'''MIT License
Copyright (C) 2020 Prokofiev Kirill, Intel Corporation
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom
the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
OR OTHER DEALINGS IN THE SOFTWARE.'''

from functools import partial

from .celeba_spoof import CelebASpoofDataset
from .lcc_fasd import LccFasdDataset


def do_nothing(**args):
    pass

# import your reader and replace do_nothing with it
external_reader=do_nothing

def get_datasets(config):

    celeba_root = config.datasets.Celeba_root
    lccfasd_root = config.datasets.LCCFASD_root


    #set of datasets
    datasets = {'celeba_spoof_train': partial(CelebASpoofDataset, root_folder=celeba_root,
                                            test_mode=0,
                                            multi_learning=config.multi_task_learning),

                'celeba_spoof_val': partial(CelebASpoofDataset, root_folder=celeba_root,
                                            test_mode=1,
                                            multi_learning=config.multi_task_learning),

                'celeba_spoof_test': partial(CelebASpoofDataset, root_folder=celeba_root,
                                            test_mode=2, multi_learning=config.multi_task_learning),
                
                'LCC_FASD_train': partial(LccFasdDataset, root_dir=lccfasd_root, protocol='train'),

                'LCC_FASD_val': partial(LccFasdDataset, root_dir=lccfasd_root, protocol='val'),

                'LCC_FASD_test': partial(LccFasdDataset, root_dir=lccfasd_root, protocol='combine_all'),

                'LCC_FASD_val_test': partial(LccFasdDataset, root_dir=lccfasd_root, protocol='val_test'),

                'LCC_FASD_combined': partial(LccFasdDataset, root_dir=lccfasd_root, protocol='combine_all'),
    }
    return datasets
