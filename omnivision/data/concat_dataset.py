# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import List, Optional, Sequence

import torch

from omnivision.data.omni_dataset import OmniDataset

from torch.utils.data import Dataset


class ConcatDataset(Dataset):
    def __init__(
        self,
        datasets: List[OmniDataset],
        max_steps: str,
        batching_strategy: str = None,
        repeat_factors: Optional[List[float]] = None,
        dataset_weights=None,
    ) -> None:
        """
        Creates an iterator that concatenates the list of datasets

        Inputs
        - dataloaders: List of dataloaders to concatenate
        - epoch: Current epoch (used for shuffling)
        - start_iter: Current iteration (unused for now)
        - concat_iterator_params: Dict containing the following
          - BATCHING_STRATEGY: String specifying batching strategy
            - "use_one": each dataloader is picked individually. Batch contains output from only one of them.
            - "use_all": outputs of all dataloaders per batch
          - MAX_STEPS: String specifying how many steps to run for
            - "sum": sum of the lengths of dataloaders. Typically used with "use_one" batching strategy
            - "max_dataset": the dataloader with the maximum length. Typically used with "use_all".
                             in this case, dataloaders with less samples can be replicated
            MAX_STEPS can also be specified as a tuple of (string, multiplier) in which case the multiplier
            scales the value specified via the string. This can control how likely it is for samples to get mixed.
          - REPEAT_FACTORS: list of ints specifying how each dataset should be repeated
                            -1 indicates that the dataset is padded to the maximum length of the underlying datasets
                            Note that in this case maximum length is *AFTER* all the datasets have been replicated.
        """
        super().__init__()
        assert isinstance(datasets, Sequence)
        self.datasets = datasets
        num_data_sources = len(self.datasets)
        self.batching_strategy = batching_strategy or "use_one"
        self.max_steps = max_steps
        self.repeat_factors = repeat_factors or [1 for _ in range(len(self.datasets))]
        assert len(self.repeat_factors) == num_data_sources

        self.dataset_weights = dataset_weights
        if self.dataset_weights is not None:
            assert sum(self.dataset_weights) == 1

    def get_loader(self, **kwargs):
        return ConcatIterator(self, **kwargs)


class ConcatIterator:
    def __init__(self, concat_dataset, **kwargs) -> None:
        assert "epoch" in kwargs
        epoch = kwargs["epoch"]
        # FIXME: this will create iters upon init instead of when iter(self) is called
        dataloaders = [x.get_loader(**kwargs) for x in concat_dataset.datasets]
        num_data_sources = len(dataloaders)
        self.iterators = [iter(x) for x in dataloaders]
        self.step_counter = 0
        self.dataloaders = dataloaders
        self.per_src_step_counter = [0 for _ in range(num_data_sources)]

        g = torch.Generator()
        g.manual_seed(epoch)

        iterator_lens = [len(x) for x in self.iterators]

        # adjust iterator lengths based on repetitions
        for idx in range(len(concat_dataset.repeat_factors)):
            # assert isinstance(
            #     concat_dataset.repeat_factors[idx], int
            # ), f"Only integer repeat factors are allowed. Found {type(concat_dataset.repeat_factors[idx])}"
            if concat_dataset.repeat_factors[idx] > 0:
                iterator_lens[idx] *= concat_dataset.repeat_factors[idx]
                iterator_lens[idx] = int(iterator_lens[idx])
            else:
                assert (
                    concat_dataset.repeat_factors[idx] == -1
                ), "repetition factor must be > 0 or -1"

        # repetition = -1
        max_dataset_len = max(iterator_lens)
        for idx in range(len(concat_dataset.repeat_factors)):
            if concat_dataset.repeat_factors[idx] == -1:
                iterator_lens[idx] = max_dataset_len

        self.iterator_lens = iterator_lens

        if isinstance(concat_dataset.max_steps, (list, tuple)):
            assert len(concat_dataset.max_steps) == 2
            max_steps_method = concat_dataset.max_steps[0]
            max_steps_mul = concat_dataset.max_steps[1]
        else:
            max_steps_method = concat_dataset.max_steps
            max_steps_mul = 1
        if max_steps_method == "sum":
            self.max_steps = int(sum(iterator_lens) * max_steps_mul)
        elif max_steps_method == "max_dataset":
            self.max_steps = int(max(iterator_lens) * max_steps_mul)
        else:
            raise ValueError(f"max_steps_method: {max_steps_method} is not supported")

        # source_to_use is a binary array of shape data_sources x steps
        # source_to_use[a, b] = 1 indicates that
        # data source `a` will be used for creating a batch at step `b`
        # this array defines how our batches are created (using one data source at a time or all or subset)

        self.source_to_use = torch.zeros(
            (num_data_sources, self.max_steps), dtype=torch.bool
        )

        if concat_dataset.batching_strategy == "use_all":
            # we first assign to every index one data source to make sure every step
            # has at least one data source
            # then we assign remaining data source ids to remaining places randomly
            for idx in range(num_data_sources):
                assert iterator_lens[idx] <= self.max_steps
            inds = torch.tensor(
                [idx for i in range(num_data_sources) for idx in [i] * iterator_lens[i]]
            )
            assert len(inds) == sum(iterator_lens)
            assert len(inds) >= self.max_steps
            inds = inds[torch.randperm(len(inds), generator=g)]
            inds_init = inds[: self.max_steps]
            for idx in range(num_data_sources):
                inds_init_ds = inds_init == idx
                self.source_to_use[idx, inds_init_ds] = 1
                # get unused indices
                inds_rem_ds = (self.source_to_use[idx] == 0).nonzero()
                # permute the indices and pick as many as we need
                rem_ds_size = iterator_lens[idx] - sum(inds_init_ds)
                inds_rem_ds = inds_rem_ds[
                    torch.randperm(len(inds_rem_ds), generator=g)
                ][:rem_ds_size]
                self.source_to_use[idx, inds_rem_ds] = 1
                assert sum(self.source_to_use[idx]) == iterator_lens[idx]

            # there should be no non-empty steps
            assert sum(self.source_to_use.sum(dim=0) == 0) == 0
        elif concat_dataset.batching_strategy == "use_one":
            if max_steps_method != "sum" and max_steps_mul != 1:
                raise NotImplementedError()

            indices = []
            for idx in range(num_data_sources):
                indices.append(torch.ones(iterator_lens[idx], dtype=torch.int64) * idx)
            indices = torch.cat(indices)
            shuffle_indices = torch.randperm(len(indices), generator=g)
            indices = indices[shuffle_indices]
            assert (
                len(indices) == self.max_steps
            ), f"Length of Indices {len(indices)} != steps {self.max_steps}"

            for idx in range(num_data_sources):
                sel_idx = torch.where(indices == idx)[0]
                self.source_to_use[idx, sel_idx] = 1

            # sanity check
            # steps should have source_to_use = 1 at every step, and it should be set for only one dataset
            assert torch.all(
                self.source_to_use.sum(dim=0)
            ).item(), "use_one logic is incorrect"
            assert self.source_to_use.sum().item() == self.max_steps

        # sanity check & logging
        logging.info(
            f"Created a ConcatIterator with batching_strategy={concat_dataset.batching_strategy} and steps {concat_dataset.max_steps}. Steps per source:"
        )
        for idx in range(num_data_sources):
            assert (
                self.source_to_use[idx].sum().item() == iterator_lens[idx]
            ), "source_to_use logic is incorrect"
            if concat_dataset.repeat_factors[idx] == -1:
                # make sure dataset is padded to max length
                assert self.source_to_use[idx].sum().item() == self.max_steps
            else:
                assert self.source_to_use[idx].sum().item() == int(
                    len(dataloaders[idx]) * concat_dataset.repeat_factors[idx]
                )
            logging.info(
                f"Dataset {idx}; len {iterator_lens[idx]}; Orig len {len(dataloaders[idx])}"
            )

    def __iter__(self):
        return self

    def get_sample(self):
        sample = {}
        """
        grad_accum_sample = False
        """
        for idx in range(len(self.iterators)):
            if not self.source_to_use[idx, self.step_counter]:
                continue
            try:
                val = next(self.iterators[idx])
            except StopIteration:
                self.iterators[idx] = iter(self.dataloaders[idx])
                val = next(self.iterators[idx])

            self.per_src_step_counter[idx] += 1
            if self.per_src_step_counter[idx] > self.iterator_lens[idx]:
                raise ValueError(
                    f"Something is off. For iterator {idx} expected {self.iterator_lens[idx]} steps but currently at {self.per_src_step_counter[idx]}"
                )

            """
            if GradAccumSampleHandler.is_grad_accum_sample(val):
                vals = GradAccumSampleHandler.strip_grad_accum_sentinel(val)
                if isinstance(sample, dict):
                    # now `sample` is a dict which we want to make a list of dictionaries
                    sample = [{x: sample[x]} for x in sample]
                sample.extend([{output_key: val} for val in vals])
                grad_accum_sample = True
            elif grad_accum_sample:
                # now `val` is a dict which we will append to the sample list
                assert isinstance(sample, list)
                sample.append({output_key: val})
            else:
                # grad accum isn't on, simply create a dictionary
                sample[output_key] = val
            """
            orig_keys = len(sample)
            sample.update(val)
            assert len(sample) == orig_keys + len(val)

        """
        if grad_accum_sample:
            return GradAccumSampleHandler.add_grad_accum_sentinel(sample)
        """
        return sample

    def __next__(self):
        if self.step_counter == self.max_steps:
            raise StopIteration

        sample = self.get_sample()

        self.step_counter += 1
        return sample

    def __len__(self):
        return self.max_steps
