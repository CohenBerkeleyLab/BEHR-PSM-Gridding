"""
make_iterable_links.py creates links to the OMI NO2 and OMPIXCOR data in the directory structure expected by the omi
package used by the PSM gridding code.
"""
from __future__ import print_function
import argparse
import datetime as dt
from glob import glob
import os
import sys

def shell_error(msg, exit_code=1):
    print(msg, file=sys.stderr)
    exit(exit_code)

def check_exec_dir():
    """
    Am I being executed in the right place? I should have OMNO2 and OMPIXCOR directories here
    :return: nothing, raises exception if check fails
    """
    if not os.path.isdir('OMNO2') or not os.path.isdir('OMPIXCOR'):
        raise RuntimeError('{} must be executed from the directory containing OMNO2 and OMPIXCOR as subdirectories'
                           .format(os.path.basename(__file__)))

def make_links(start_date, end_date, dry_run=False):
    curr_date = start_date
    while curr_date <= end_date:
        _make_product_links(curr_date, 'OMNO2', 'OMNO2.003', 'OMI-Aura_L2-OMNO2_', dry_run=dry_run)
        _make_product_links(curr_date, 'OMPIXCOR', 'OMPIXCOR.003', 'OMI-Aura_L2-OMPIXCOR_', dry_run=dry_run)
        curr_date += dt.timedelta(days=1)

def _make_product_links(curr_date, current_top, new_top, file_stem, dry_run):
    year_str = curr_date.strftime('%Y')
    month_str = curr_date.strftime('%m')
    doy_str = curr_date.strftime('%j')
    omi_date_str = curr_date.strftime('%Ym%m%d')

    # Find all the existing files for the current day
    extant_path = os.path.join(current_top, year_str, month_str, '{}{}*.he5'.format(file_stem, omi_date_str))
    extant_files = glob(extant_path)

    # Put them in a subdirectory by day-of-year. Make any intermediate directories required. 
    new_path = os.path.join('PSM_Links', new_top, 'level2', year_str, doy_str)
    back_out = os.path.join('..','..','..','..','..')

    for f in extant_files:
        base_name = os.path.basename(f)
        link_name = os.path.join(back_out, f)
        if dry_run:
            print('{} -> {} in {}'.format(link_name, base_name, os.path.join(os.getcwd(), new_path)))
        else:
            if not os.path.isdir(new_path):
                os.makedirs(new_path)
            os.chdir(new_path)
            os.symlink(link_name, base_name)
            os.chdir(back_out)




def get_args():
    parser = argparse.ArgumentParser(description='Create the required directory structure for the omi.iter_orbits function')
    parser.add_argument('start_date', help='First date to link, in yyyy-mm-dd format')
    parser.add_argument('end_date', help='Last date to link, in yyyy-mm-dd format')
    parser.add_argument('-d', '--dry-run', action='store_true', help='Just print the links that will be made, don\'t make them')

    args = parser.parse_args()
    try:
        start_date = dt.datetime.strptime(args.start_date, '%Y-%m-%d')
    except ValueError:
        shell_error('Start date is not in yyyy-mm-dd format')

    try:
        end_date = dt.datetime.strptime(args.end_date, '%Y-%m-%d')
    except ValueError:
        shell_error('End date is not in yyyy-mm-dd format')

    return start_date, end_date, args.dry_run

if __name__ == '__main__':
    start_date, end_date, dry_run = get_args()
    make_links(start_date, end_date, dry_run=dry_run)
