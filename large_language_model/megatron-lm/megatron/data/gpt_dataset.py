# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GPT style dataset."""

import os
import time
import math

import numpy as np
import torch

from megatron import mpu, print_rank_0
from megatron.data.blendable_dataset import BlendableDataset
from megatron.data.dataset_utils import get_datasets_weights_and_num_samples
from megatron.data.dataset_utils import get_train_valid_test_split_
from megatron.data.indexed_dataset import make_dataset as make_indexed_dataset


def build_train_valid_test_datasets(data_prefix, train_data_prefix, 
                                    valid_data_prefix, test_data_prefix, 
                                    data_impl, splits_string,
                                    train_valid_test_num_samples,
                                    seq_length, seed, skip_warmup,
                                    use_seq_len_plus_one_tokens=True):
    """Build train, valid, and test datasets."""

    if data_prefix:
        print_rank_0("Single data path provided for train, valid & test")
        # Single dataset.
        if len(data_prefix) == 1:
            return _build_train_valid_test_datasets(data_prefix[0],
                                                    data_impl, splits_string,
                                                    train_valid_test_num_samples,
                                                    seq_length, seed, skip_warmup)

        # Blending dataset.
        # Parse the values.
        output = get_datasets_weights_and_num_samples(data_prefix,
                                                    train_valid_test_num_samples)
        prefixes, weights, datasets_train_valid_test_num_samples = output

        # Build individual datasets.
        train_datasets = []
        valid_datasets = []
        test_datasets = []
        for i in range(len(prefixes)):
            train_ds, valid_ds, test_ds = _build_train_valid_test_datasets(
                prefixes[i], data_impl, splits_string,
                datasets_train_valid_test_num_samples[i],
                seq_length, seed, skip_warmup,
                use_seq_len_plus_one_tokens)
            if train_ds:
                train_datasets.append(train_ds)
            if valid_ds:
                valid_datasets.append(valid_ds)
            if test_ds:
                test_datasets.append(test_ds)

        # Blend.
        blending_train_dataset = None
        if train_datasets:
            blending_train_dataset = BlendableDataset(train_datasets, weights)
        blending_valid_dataset = None
        if valid_datasets:
            blending_valid_dataset = BlendableDataset(valid_datasets, weights)
        blending_test_dataset = None
        if test_datasets:
            blending_test_dataset = BlendableDataset(test_datasets, weights)

        return (blending_train_dataset, blending_valid_dataset,
                blending_test_dataset)
    else:
        print_rank_0("Separate data paths provided for train, valid & test. Split string will be ignored.")
        assert (train_data_prefix is not None)
        train_dataset, valid_dataset, test_dataset = None, None, None
        # Single dataset.
        train_dataset = build_dataset("train", train_data_prefix, data_impl,
                                    train_valid_test_num_samples[0], seq_length, seed,
                                    skip_warmup, use_seq_len_plus_one_tokens)

        if valid_data_prefix is not None:
            valid_dataset = build_dataset("valid", valid_data_prefix, data_impl,
                                    train_valid_test_num_samples[1], seq_length, seed,
                                    False, use_seq_len_plus_one_tokens)

        if test_data_prefix is not None:
            test_dataset = build_dataset("test", test_data_prefix, data_impl,
                                    train_valid_test_num_samples[2], seq_length, seed,
                                    False, use_seq_len_plus_one_tokens)

        return (train_dataset, valid_dataset, test_dataset)


def build_dataset(dataset_name, data_prefix, data_impl, num_samples, seq_length, seed, skip_warmup, use_seq_len_plus_one_tokens):
    dataset = None
    if len(data_prefix) == 1:
        dataset = _build_dataset(dataset_name,
                        data_prefix[0], data_impl,
                        num_samples, seq_length,
                        seed, skip_warmup,
                        use_seq_len_plus_one_tokens)
    else:
        # Blending dataset.
        # Parse the values.
        output = get_datasets_weights_and_num_samples(data_prefix, num_samples)
        prefixes, weights, dataset_num_samples = output

        # Build individual datasets.
        datasets = []
        for i in range(len(prefixes)):
            ds = _build_dataset(dataset_name, prefixes[i],
                            data_impl, dataset_num_samples[i],
                            seq_length, seed, skip_warmup,
                            use_seq_len_plus_one_tokens)
            if ds:
                datasets.append(ds)

        if datasets:
            dataset = BlendableDataset(datasets, weights)

    return dataset


def _build_dataset(dataset_name, data_prefix, data_impl,
                num_samples, seq_length, seed, skip_warmup,
                use_seq_len_plus_one_tokens):
    """
    Build dataset. This method is called when individual
    train, valid, test datasets are provided
    """

    # Indexed dataset.
    indexed_dataset = get_indexed_dataset_(data_prefix,
                                           data_impl,
                                           skip_warmup)

    total_num_of_documents = indexed_dataset.sizes.shape[0]

    print_rank_0('    {}:'.format(dataset_name))
    print_rank_0('     document indices in [0, {}) total of {} '
                 'documents'.format(total_num_of_documents, total_num_of_documents))

    documents = np.arange(start=0, stop=total_num_of_documents,
                        step=1, dtype=np.int32)

    dataset = GPTDataset(dataset_name, data_prefix,
                        documents, indexed_dataset,
                        num_samples, seq_length, seed,
                        use_seq_len_plus_one_tokens)

    return dataset


def _build_train_valid_test_datasets(data_prefix, data_impl, splits_string,
                                     train_valid_test_num_samples,
                                     seq_length, seed, skip_warmup,
                                     use_seq_len_plus_one_tokens):
    """Build train, valid, and test datasets."""

    # Indexed dataset.
    indexed_dataset = get_indexed_dataset_(data_prefix,
                                           data_impl,
                                           skip_warmup)

    total_num_of_documents = indexed_dataset.sizes.shape[0]
    splits = get_train_valid_test_split_(splits_string, total_num_of_documents)

    # Print stats about the splits.
    print_rank_0(' > dataset split:')

    def print_split_stats(name, index):
        print_rank_0('    {}:'.format(name))
        print_rank_0('     document indices in [{}, {}) total of {} '
                     'documents'.format(splits[index], splits[index + 1],
                                        splits[index + 1] - splits[index]))
    print_split_stats('train', 0)
    print_split_stats('validation', 1)
    print_split_stats('test', 2)

    def build_dataset(index, name):
        dataset = None
        if splits[index + 1] > splits[index]:
            documents = np.arange(start=splits[index], stop=splits[index + 1],
                                  step=1, dtype=np.int32)
            dataset = GPTDataset(name, data_prefix,
                                  documents, indexed_dataset,
                                  train_valid_test_num_samples[index],
                                  seq_length, seed,
                                  use_seq_len_plus_one_tokens)
        return dataset

    train_dataset = build_dataset(0, 'train')
    valid_dataset = build_dataset(1, 'valid')
    test_dataset = build_dataset(2, 'test')

    return (train_dataset, valid_dataset, test_dataset)


def get_indexed_dataset_(data_prefix, data_impl, skip_warmup):
    """Build indexed dataset."""
    print_rank_0(' > building dataset index ...')

    start_time = time.time()
    indexed_dataset = make_indexed_dataset(data_prefix,
                                           data_impl,
                                           skip_warmup)
    print_rank_0(' > finished creating indexed dataset in {:4f} '
                 'seconds'.format(time.time() - start_time))
    print_rank_0('    number of documents: {}'.format(
        indexed_dataset.sizes.shape[0]))

    return indexed_dataset


class GPTDataset(torch.utils.data.Dataset):

    def __init__(self, name, data_prefix, documents, indexed_dataset,
                 num_samples, seq_length, seed, use_seq_len_plus_one_tokens):

        self.name = name
        self.indexed_dataset = indexed_dataset
        self.seq_length = seq_length
        self.add_extra_token = 0
        if use_seq_len_plus_one_tokens:
            self.add_extra_token = 1

        # Checks
        assert np.min(documents) >= 0
        assert np.max(documents) < indexed_dataset.sizes.shape[0]

        # Build index mappings.
        self.doc_idx, self.sample_idx, self.shuffle_idx = _build_index_mappings(
            self.name, data_prefix, documents, self.indexed_dataset.sizes,
            num_samples, seq_length, seed, self.add_extra_token)

    def __len__(self):
        # -1 is due to data structure used to retieve the index:
        #    sample i --> [sample_idx[i], sample_idx[i+1])
        return self.sample_idx.shape[0] - 1

    def __getitem__(self, idx):
        # Get the shuffled index.
        dummy_sample = idx < 0
        idx = np.abs(idx)
        idx = self.shuffle_idx[idx]
        # Start and end documents and offsets.
        doc_index_f = self.sample_idx[idx][0]
        doc_index_l = self.sample_idx[idx + 1][0]
        offset_f = self.sample_idx[idx][1]
        offset_l = self.sample_idx[idx + 1][1]
        # If we are within the same document, just extract the chunk.
        if doc_index_f == doc_index_l:
            sample = self.indexed_dataset.get(self.doc_idx[doc_index_f],
                                              offset=offset_f,
                                              length=offset_l - offset_f + self.add_extra_token)
        else:
            # Otherwise, get the rest of the initial document.
            sample_list = [self.indexed_dataset.get(self.doc_idx[doc_index_f],
                                                    offset=offset_f)]
            # Loop over all in between documents and add the entire document.
            for i in range(doc_index_f + 1, doc_index_l):
                sample_list.append(self.indexed_dataset.get(self.doc_idx[i]))
            # And finally add the relevant portion of last document.
            sample_list.append(self.indexed_dataset.get(
                self.doc_idx[doc_index_l],
                length=offset_l + self.add_extra_token))
            sample = np.concatenate(sample_list)
        if len(sample) != (self.seq_length + self.add_extra_token):
            sample = np.array(sample, dtype=np.int64)
            sample = np.pad(sample, (0, self.seq_length + self.add_extra_token - len(sample)), mode='constant', constant_values=-1)
        return {'text': np.array(sample, dtype=np.int64),
                'dummy_sample': np.array(int(dummy_sample), dtype=np.int64)}


def _build_index_mappings(name, data_prefix, documents, sizes,
                          num_samples, seq_length, seed, add_extra_token):
    """Build doc-idx, sample-idx, and shuffle-idx.
    doc-idx: is an array (ordered) of documents to be used in training.
    sample-idx: is the start document index and document offset for each
       training sample.
    shuffle-idx: maps the sample index into a random index into sample-idx.
    """
    # Number of tokens in each epoch and number of required epochs.
    tokens_per_epoch = _num_tokens(documents, sizes)
    num_epochs = _num_epochs(tokens_per_epoch, seq_length, num_samples, add_extra_token)
    if num_samples < 0:
        print_num_samples = tokens_per_epoch // seq_length
    else:
        print_num_samples = num_samples
    # rng state
    np_rng = np.random.RandomState(seed=seed)

    # Filename of the index mappings.
    _filename = data_prefix
    _filename += '_{}_indexmap'.format(name)
    _filename += '_{}ns'.format(print_num_samples)
    _filename += '_{}sl'.format(seq_length)
    _filename += '_{}s'.format(seed)
    doc_idx_filename = _filename + '_doc_idx.npy'
    sample_idx_filename = _filename + '_sample_idx.npy'
    shuffle_idx_filename = _filename + '_shuffle_idx.npy'

    # Build the indexed mapping if not exist.
    if torch.distributed.get_rank() == 0:
        if (not os.path.isfile(doc_idx_filename)) or \
           (not os.path.isfile(sample_idx_filename)) or \
           (not os.path.isfile(shuffle_idx_filename)):

            print_rank_0(' > WARNING: could not find index map files, building '
                         'the indices on rank 0 ...')

            # For the last epoch, decide whether include the entire epoch
            # in the global shuffle or not.

            # If we need only one epoch, then separating last epoch  does
            # not mean anything.
            if num_epochs == 1:
                separate_last_epoch = False
                print(' > only one epoch required, setting '
                      'separate_last_epoch to False', flush=True)

            else:
                # Get the number of samples for the last epoch
                assert num_samples >= 0, 'number of samples should be non-negative'
                num_samples_from_epochs_minus_one = (
                    (num_epochs - 1) * tokens_per_epoch - add_extra_token) // seq_length
                last_epoch_num_samples = num_samples - \
                                         num_samples_from_epochs_minus_one
                assert last_epoch_num_samples >= 0, \
                    'last epoch number of samples should be non-negative.'
                num_samples_per_epoch = (tokens_per_epoch - add_extra_token) // seq_length
                assert last_epoch_num_samples < (num_samples_per_epoch + 1), \
                    'last epoch number of samples exceeded max value.'
                # If we have less than 80% of the samples for the last epoch,
                # seperate out the epoch and treat it differently.
                # Note: the 80% number is just based on common sense and can
                # be adjusted if needed.
                separate_last_epoch = (last_epoch_num_samples <
                                       int(0.80 * num_samples_per_epoch))
                if separate_last_epoch:
                    string = ' > last epoch number of samples ({}) is smaller '\
                             'than 80% of number of samples per epoch ({}), '\
                             'setting separate_last_epoch to True'
                else:
                    string = ' > last epoch number of samples ({}) is larger '\
                             'than 80% of number of samples per epoch ({}), '\
                             'setting separate_last_epoch to False'
                print(string.format(last_epoch_num_samples,
                                    num_samples_per_epoch), flush=True)

            # doc-idx.
            start_time = time.time()
            doc_idx = _build_doc_idx(documents, num_epochs, np_rng,
                                     separate_last_epoch)
            np.save(doc_idx_filename, doc_idx, allow_pickle=True)
            print_rank_0(' > elasped time to build and save doc-idx mapping '
                         '(seconds): {:4f}'.format(time.time() - start_time))
            # sample-idx.
            start_time = time.time()
            # Use C++ implementation for speed.
            # First compile and then import.
            from megatron.data import helpers
            assert doc_idx.dtype == np.int32
            assert sizes.dtype == np.int32
            sample_idx = helpers.build_sample_idx(sizes, doc_idx, seq_length,
                                                  num_epochs, tokens_per_epoch,
                                                  num_samples < 0, add_extra_token)
            # sample_idx = _build_sample_idx(sizes, doc_idx, seq_length,
            #                              num_epochs, tokens_per_epoch,
            #                              num_samples < 0, add_extra_token)
            np.save(sample_idx_filename, sample_idx, allow_pickle=True)
            print_rank_0(' > elasped time to build and save sample-idx mapping '
                         '(seconds): {:4f}'.format(time.time() - start_time))
            # shuffle-idx.
            start_time = time.time()
            # -1 is due to data structure used to retieve the index:
            #    sample i --> [sample_idx[i], sample_idx[i+1])
            if separate_last_epoch:
                num_samples_ = num_samples_from_epochs_minus_one
            else:
                num_samples_ = sample_idx.shape[0] - 1
            shuffle_idx = _build_shuffle_idx(num_samples_,
                                             sample_idx.shape[0] - 1, np_rng)
            np.save(shuffle_idx_filename, shuffle_idx, allow_pickle=True)
            print_rank_0(' > elasped time to build and save shuffle-idx mapping'
                         ' (seconds): {:4f}'.format(time.time() - start_time))

    # This should be a barrier but nccl barrier assumes
    # device_index=rank which is not the case for model
    # parallel case
    counts = torch.cuda.LongTensor([1])
    torch.distributed.all_reduce(counts, group=mpu.get_data_parallel_group())
    torch.distributed.all_reduce(counts, group=mpu.get_pipeline_model_parallel_group())
    assert counts[0].item() == (
        torch.distributed.get_world_size() //
        torch.distributed.get_world_size(group=mpu.get_tensor_model_parallel_group()))

    # Load mappings.
    start_time = time.time()
    print_rank_0(' > loading doc-idx mapping from {}'.format(
        doc_idx_filename))
    doc_idx = np.load(doc_idx_filename, allow_pickle=True, mmap_mode='r')
    print_rank_0(' > loading sample-idx mapping from {}'.format(
        sample_idx_filename))
    sample_idx = np.load(sample_idx_filename, allow_pickle=True, mmap_mode='r')
    print_rank_0(' > loading shuffle-idx mapping from {}'.format(
        shuffle_idx_filename))
    shuffle_idx = np.load(shuffle_idx_filename, allow_pickle=True, mmap_mode='r')
    print_rank_0('    loaded indexed file in {:3.3f} seconds'.format(
        time.time() - start_time))
    print_rank_0('    total number of samples: {}'.format(
        sample_idx.shape[0]))
    print_rank_0('    total number of epochs: {}'.format(num_epochs))

    return doc_idx, sample_idx, shuffle_idx


def _num_tokens(documents, sizes):
    """Total number of tokens in the dataset."""
    return np.sum(sizes[documents])


def _num_epochs(tokens_per_epoch, seq_length, num_samples, add_extra_token):
    """Based on number of samples and sequence lenght, calculate how many
    epochs will be needed."""
    num_epochs = 0
    total_tokens = 0
    while True:
        num_epochs += 1
        total_tokens += tokens_per_epoch
        # -1 is because we need to retrieve seq_length + 1 token each time
        # but the last token will overlap with the first token of the next
        # sample except for the last sample.
        if ((total_tokens - add_extra_token) // seq_length) >= num_samples:
            return num_epochs


def _build_doc_idx(documents, num_epochs, np_rng, separate_last_epoch):
    """Build an array with length = number-of-epochs * number-of-dcuments.
    Each index is mapped to a corresponding document."""
    if not separate_last_epoch or num_epochs == 1:
        doc_idx = np.mgrid[0:num_epochs, 0:len(documents)][1]
        doc_idx[:] = documents
        doc_idx = doc_idx.reshape(-1)
        doc_idx = doc_idx.astype(np.int32)
        np_rng.shuffle(doc_idx)
        return doc_idx

    doc_idx_first = _build_doc_idx(documents, num_epochs-1, np_rng, False)
    doc_idx_last = _build_doc_idx(documents, 1, np_rng, False)
    return np.concatenate((doc_idx_first, doc_idx_last))


def _build_sample_idx(sizes, doc_idx, seq_length,
                      num_epochs, tokens_per_epoch,
                      keep_last_sequence, add_extra_token):
    """Sample index mapping is a 2D array with sizes
    [number-of-samples + 1, 2] where [..., 0] contains
    the index into `doc_idx` and [..., 1] is the
    starting offset in that document."""

    # Total number of samples. For -1 see comments in `_num_epochs`.
    if keep_last_sequence:
        num_samples = math.ceil((num_epochs * tokens_per_epoch - add_extra_token) / seq_length)
    else:
        num_samples = (num_epochs * tokens_per_epoch - add_extra_token) // seq_length
    #befor_num_samples = (num_epochs * tokens_per_epoch - 1) // seq_length
    #print(F"Num Epochs: {num_epochs}, Tokens per epoch: {tokens_per_epoch}, num_samples: {num_samples}, before num_samples: {befor_num_samples}")
    #import pdb; pdb.set_trace()
    sample_idx = np.zeros([num_samples + 1, 2], dtype=np.int32)

    # Index into sample_idx.
    sample_index = 0
    # Index into doc_idx.
    doc_idx_index = 0
    # Begining offset for each document.
    doc_offset = 0
    # Start with first document and no offset.
    sample_idx[sample_index][0] = doc_idx_index
    sample_idx[sample_index][1] = doc_offset
    sample_index += 1
    while sample_index <= num_samples:
        # Start with a fresh sequence.
        remaining_seq_length = seq_length + add_extra_token
        while remaining_seq_length != 0:
            # Get the document length.
            doc_id = doc_idx[doc_idx_index]
            doc_length = sizes[doc_id] - doc_offset
            # And add it to the current sequence.
            remaining_seq_length -= doc_length
            # If we have more than a full sequence, adjust offset and set
            # remaining length to zero so we return from the while loop.
            # Note that -1 here is for the same reason we have -1 in
            # `_num_epochs` calculations.
            if remaining_seq_length <= 0:
                doc_offset += (remaining_seq_length + doc_length - add_extra_token)
                remaining_seq_length = 0
            else:
                # Otherwise, start from the begining of the next document.
                if doc_idx_index == (len(doc_idx) - 1):
                    assert sample_index == num_samples, F"sample_index={sample_index} and num_samples={num_samples} should be the same"
                    doc_offset = sizes[doc_idx[doc_idx_index]] - add_extra_token
                    break
                doc_idx_index += 1
                doc_offset = 0
        # Record the sequence.
        sample_idx[sample_index][0] = doc_idx_index
        sample_idx[sample_index][1] = doc_offset
        sample_index += 1

    return sample_idx


def _build_shuffle_idx(num_samples, total_size, np_rng):
    """Build the range [0, size) and shuffle."""
    print(' > building shuffle index with split [0, {}) and [{}, {}) '
          '...'.format(num_samples, num_samples, total_size), flush=True)
    
    dtype_ = np.uint32
    if total_size >= (np.iinfo(np.uint32).max - 1):
        dtype_ = np.int64

    shuffle_idx_first = np.arange(start=0, stop=num_samples,
                                  step=1, dtype=dtype_)
    np_rng.shuffle(shuffle_idx_first)
    if num_samples == total_size:
        return shuffle_idx_first

    shuffle_idx_last = np.arange(start=num_samples, stop=total_size,
                                 step=1, dtype=dtype_)
    np_rng.shuffle(shuffle_idx_last)

    return np.concatenate((shuffle_idx_first, shuffle_idx_last))
