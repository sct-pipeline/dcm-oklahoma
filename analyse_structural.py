#!/usr/bin/env python
# -*- coding: utf-8

# Analyses results of anatomical data

# Author: Sandrine Bédard

import os
import re
import logging
import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import scipy.stats as stats
import yaml


FNAME_LOG = 'log_stats.txt'

# Initialize logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # default: logging.DEBUG, logging.INFO
hdlr = logging.StreamHandler(sys.stdout)
logging.root.addHandler(hdlr)


METRICS = ['MEAN(area)', 'MEAN(diameter_AP)', 'MEAN(diameter_RL)', 'MEAN(eccentricity)',
           'MEAN(solidity)']


METRICS_TO_YLIM = {
    'MEAN(diameter_AP)': (4, 9.3),
    'MEAN(area)': (30, 95),
    'MEAN(diameter_RL)': (8.5, 16),
    'MEAN(eccentricity)': (0.6, 0.95),
    'MEAN(solidity)': (0.912, 0.999),
}


METRIC_TO_AXIS = {
    'MEAN(diameter_AP)': 'AP Diameter [mm]',
    'MEAN(area)': 'Cross-Sectional Area [mm²]',
    'MEAN(diameter_RL)': 'Transverse Diameter [mm]',
    'MEAN(eccentricity)': 'Eccentricity [a.u.]',
    'MEAN(solidity)': 'Solidity [%]',
}


PALETTE = {
    'sex': {'M': 'blue', 'F': 'red'},
    'group': {'HC': 'blue', 'CS': '#e31a1c'}
    }

LABELS_FONT_SIZE = 14
TICKS_FONT_SIZE = 12


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i-folder",
                        required=True,
                        type=str,
                        help="Results folder of spinal cord preprocessing")
    parser.add_argument("-o-folder",
                        type=str,
                        required=True,
                        help="Folder to right results")
    return parser


def fetch_participant_session_and_group(filename_path):
    """
    Get participant_id, session_id, and group (CSM/HC) from the file path

    If the filename_path DOES NOT contain 'f', it means that it represents ses-01
    If the filename_path contains 'f', it means that it represents ses-02
    If the filename_path contains 'f2', it means that it represents ses-03

    :param filename_path: e.g., '~/data/dcm-oklahoma/Long_Study_Results/sub-CSM048f/ses-spinalcord/anat/T2w/t2w_shape_PAM50_perslice.csv''
    :return: participant_id: e.g., 'sub-CSM048'
    :return: session_id: e.g., ses-01
    :return: group: e.g., 'CSM' or 'HC'
    """

    participant_tmp = re.search('sub-(.*?)[_/]', filename_path)     # [_/] slash or underscore
    participant_id = participant_tmp.group(0)[:-1] if participant_tmp else ""    # [:-1] removes the last underscore or slash
    # REGEX explanation
    # . - match any character (except newline)
    # *? - match the previous element as few times as possible (zero or more times)

    # Fetch session_id from participant_id (see the function docstring for more details)
    if 'f2' in participant_id:
        session_id = 'ses-03'
    elif 'f' in participant_id:
        session_id = 'ses-02'
    else:
        session_id = 'ses-01'

    # Now, we can remove 'f' and 'f2' from the participant_id
    participant_id = participant_id.replace('f2', '').replace('f', '')

    # Fetch group
    if 'CSM' in participant_id:
        group = 'CSM'
    elif 'HC' in participant_id:
        group = 'HC'

    return participant_id, session_id, group


def format_pvalue(p_value, alpha=0.001, decimal_places=3, include_space=True, include_equal=True):
    """
    Format p-value.
    If the p-value is lower than alpha, format it to "<0.001", otherwise, round it to three decimals

    :param p_value: input p-value as a float
    :param alpha: significance level
    :param decimal_places: number of decimal places the p-value will be rounded
    :param include_space: include space or not (e.g., ' = 0.06')
    :param include_equal: include equal sign ('=') to the p-value (e.g., '=0.06') or not (e.g., '0.06')
    :return: p_value: the formatted p-value (e.g., '<0.05') as a str
    """
    if include_space:
        space = ' '
    else:
        space = ''

    # If the p-value is lower than alpha, return '<alpha' (e.g., <0.001)
    if p_value < alpha:
        p_value = space + "<" + space + str(alpha)
    # If the p-value is greater than alpha, round it number of decimals specified by decimal_places
    else:
        if include_equal:
            p_value = space + '=' + space + str(round(p_value, decimal_places))
        else:
            p_value = space + str(round(p_value, decimal_places))

    return p_value


def compare_metrics_across_group(df, perlevel=False, metric_chosen=None):
    """
    Compute Wilcoxon rank-sum tests between males and females for each metric.
    """

    logger.info("")

    for metric in METRICS:
        logger.info(f"\n{metric}")
        if metric_chosen:
            metric=metric_chosen
        if perlevel:
            slices_HC = df[df['group'] == 'HC'].groupby(['VertLevel'])[metric].mean()
            slices_HC_STD = df[df['group'] == 'HC'].groupby(['VertLevel'])[metric].std()
            slices_CR = df[df['group'] == 'CR'].groupby(['VertLevel'])[metric].mean()
            slices_CR_STD = df[df['group'] == 'CR'].groupby(['VertLevel'])[metric].std()
            logger.info(f'Mean {metric} for HC: {slices_HC}')
            logger.info(f'STD {metric} for HC: {slices_HC_STD}')
            logger.info(f'Mean {metric} for CR: {slices_CR}')
            logger.info(f'STD {metric} for CR: {slices_CR_STD}')
        else:

            # Get mean values for each slice
            slices_HC = df[df['group'] == 'HC'].groupby(['Slice (I->S)'])[metric].mean()
            slices_CR = df[df['group'] == 'CR'].groupby(['Slice (I->S)'])[metric].mean()

        # Run normality test
        stat, pval = stats.shapiro(slices_HC)
        logger.info(f'Normality test HC: p-value{format_pvalue(pval)}')
        stat, pval = stats.shapiro(slices_CR)
        logger.info(f'Normality test CR: p-value{format_pvalue(pval)}')
        # Run Wilcoxon rank-sum test (groups are independent)
        from statsmodels.sandbox.stats.multicomp import multipletests
        stat, pval = stats.ranksums(x=slices_HC, y=slices_CR)
        #p_adjusted = multipletests(pval, method='bonferroni')
        #print(p_adjusted)
        logger.info(f'{metric}: Wilcoxon rank-sum test between HC and CR: p-value{format_pvalue(pval)}')
        if metric_chosen:
           break


def get_vert_indices(df, vertlevel='VertLevel'):
    """
    Get indices of slices corresponding to mid-vertebrae
    Args:
        df (pd.dataFrame): dataframe with CSA values
    Returns:
        vert (pd.Series): vertebrae levels across slices
        ind_vert (np.array): indices of slices corresponding to the beginning of each level (=intervertebral disc)
        ind_vert_mid (np.array): indices of slices corresponding to mid-levels
    """
    # Get unique participant IDs
    subjects = df['participant_id'].unique()
    # Get vert levels for one certain subject
    vert = df[(df['participant_id'] == subjects[0]) & (df['session'] == 'ses-01')][vertlevel]
    # Get indexes of where array changes value
    ind_vert = vert.diff()[vert.diff() != 0].index.values
    # Get the beginning of C1
    ind_vert = np.append(ind_vert, vert.index.values[-1])
    ind_vert_mid = []
    # Get indexes of mid-vertebrae
    for i in range(len(ind_vert)-1):
        ind_vert_mid.append(int(ind_vert[i:i+2].mean()))

    return vert, ind_vert, ind_vert_mid


def read_t2w_pam50(folder):
    """
    Read CSV files with morphometrics normalized to PAM50 space
    """
    # Get recursively list of all t2w_shape_PAM50_perslice.csv files
    files_list = [os.path.join(root, file) for root, _, files in os.walk(folder) for file in files if
                  't2w_shape_PAM50_perslice.csv' in file]

    # Read all CSV files and concatenate them
    combined_df = pd.DataFrame()
    for file in files_list:
        df = pd.read_csv(os.path.join(folder, file))
        combined_df = pd.concat([combined_df, df], ignore_index=True)

    # Fetch participant and session using lambda function
    combined_df['participant_id'], combined_df['session'], combined_df['group'] = (
        zip(*combined_df['Filename'].map(lambda x: fetch_participant_session_and_group(x))))

    # Keep only relevant columns
    combined_df = combined_df[['participant_id', 'session', 'group', 'VertLevel', 'Slice (I->S)',
                               'MEAN(area)', 'MEAN(diameter_AP)', 'MEAN(diameter_RL)', 'MEAN(eccentricity)',
                               'MEAN(solidity)']].drop(0)
    return combined_df


def read_t2w_pam50_old(fname, suffix='_T2w_seg.nii.gz', session=None, exclude_list=None):
    data = pd.read_csv(fname)
    # Filter with session first
    data['participant_id'] = (data['Filename'].str.split('/').str[-1]).str.replace(suffix, '')
    data['session'] = data['participant_id'].str.split('_').str[-1]
    data['group'] = data['participant_id'].str.split('_').str[-2].str.split('-').str[-1].str[0:2]
    # Drop subjects with id
    if exclude_list:
        for subject in exclude_list:
            sub = (subject+'_'+ session)
            logger.info(f'dropping {sub}')
            data = data.drop(data[data['participant_id'] == sub].index, axis=0)
    return data.loc[data['session'] == session]




def get_number_subjects(df, session):

    list_subject = np.unique(df['participant_id'].to_list())
    list_session = [subject for subject in list_subject if session in subject]
    nb_subjects_session = len(list_session)
    nb_subject_CR = len([subject for subject in list_session if 'CR' in subject])
   # print([subject for subject in list_session if 'CR' in subject])

    nb_subject_HC = len([subject for subject in list_session if 'HC' in subject])
   # print([subject for subject in list_session if 'HC' in subject])
    logger.info(f'Total number of subject for {session}: {nb_subjects_session}')
    logger.info(f'With CR = {nb_subject_CR} and HC = {nb_subject_HC}')


def plot_dice(df, hue, metric, path_out, filename):
    plt.figure()
    fig, ax = plt.subplots()
    sns.lineplot(ax=ax, data=df, x="Slice (I->S)", y=metric, errorbar='sd', hue=hue, linewidth=2,
                 palette=PALETTE[hue])
    ymin, ymax = ax.get_ylim()
    # Get indices of slices corresponding vertebral levels
    vert, ind_vert, ind_vert_mid = get_vert_indices(df, vertlevel='VertLevel')
    # Insert a vertical line for each intervertebral disc
    for idx, x in enumerate(ind_vert[1:-1]):
        ax.axvline(df.loc[x, 'Slice (I->S)'], color='black', linestyle='--', alpha=0.5, zorder=0)
    # Insert a text label for each vertebral level
    for idx, x in enumerate(ind_vert_mid, 0):
        # Deal with T1 label (C8 -> T1)
        if vert[x] > 7:
            level = 'T' + str(vert[x] - 7)
            ax.text(df.loc[ind_vert_mid[idx], 'Slice (I->S)'], ymin, level, horizontalalignment='center',
                            verticalalignment='bottom', color='black', fontsize=TICKS_FONT_SIZE)
        else:
            level = 'C' + str(vert[x])
            ax.text(df.loc[ind_vert_mid[idx], 'Slice (I->S)'], ymin, level, horizontalalignment='center',
                            verticalalignment='bottom', color='black', fontsize=TICKS_FONT_SIZE)

    # Invert x-axis
    ax.invert_xaxis()
    # Add only horizontal grid lines
    ax.yaxis.grid(True)
    # Move grid to background (i.e. behind other elements)
    ax.set_axisbelow(True)    

    # Save figure
    path_filename = os.path.join(path_out, filename)
    plt.savefig(path_filename, dpi=500, bbox_inches='tight')
    logger.info('Figure saved: ' + path_filename)


def avg_metrics(df, nb_slices=8, vertlevel='VertLevel', metric=None):
    # Get indices of slices corresponding vertebral levels
    vert, ind_vert, ind_vert_mid = get_vert_indices(df, vertlevel=vertlevel)
    Levels = [3, 4, 5]
    vert_unique = np.unique(vert)[::-1]
    df_avg = pd.DataFrame()
    j=0
    list_participants = np.unique(df['participant_id'])
    participants = []
    groups = []
    vertlevels = []
    sessions = []
    for metric_id in METRICS:
        total_metric_mean = [] 

        print(metric_id)
        if metric:
            metric_id = metric
        for level in Levels:
            for participant in list_participants:
                idx = np.argwhere(vert_unique==level)
                slice_nb_mid = df.loc[ind_vert[idx][0][0], 'Slice (I->S)']
                slice_nb_min = slice_nb_mid - nb_slices//2
                slice_nb_max = slice_nb_mid + nb_slices//2
                # Get dataframe of single participant
                df_participant = df.loc[df['participant_id']==participant]
                metric_mean = 0
                # Fetch metric
                i=0
                for slice_nb in range(slice_nb_min, slice_nb_max + 1):
                    metric_mean += df_participant.loc[df_participant['Slice (I->S)']==slice_nb, metric_id].values[0]
                    i+=1
                # Average metric
                metric_mean = metric_mean/i
                total_metric_mean.append(metric_mean)
                if j < 1:
                    participants.append(participant)
                    vertlevels.append(level)
                    groups.append(np.unique(df_participant['group'])[0])
                    sessions.append(np.unique(df_participant['session'])[0])
        df_avg[metric_id] = total_metric_mean
        j+=1
        if metric:
            break
    df_avg['participant_id'] = participants
    df_avg['VertLevel'] = vertlevels
    df_avg['session'] = sessions
    df_avg['group'] = groups
    return df_avg

def create_lineplot(df, hue, filename=None):
    """
    Create lineplot for individual metrics per vertebral levels.
    Note: we are ploting slices not levels to avoid averaging across levels.
    Args:
        df (pd.dataFrame): dataframe with metric values
        hue (str): column name of the dataframe to use for grouping; if None, no grouping is applied
    """

    #mpl.rcParams['font.family'] = 'Arial'

    fig, axes = plt.subplots(1, 5, figsize=(25, 4))
    axs = axes.ravel()

    # Loop across metrics
    for index, metric in enumerate(METRICS):
        # Note: we are ploting slices not levels to avoid averaging across levels
        if hue == 'sex' or hue=='group':
            sns.lineplot(ax=axs[index], x="Slice (I->S)", y=metric, data=df, errorbar='sd', hue=hue, linewidth=2,
                         palette=PALETTE[hue])
            if index == 0:
                axs[index].legend(loc='upper right', fontsize=TICKS_FONT_SIZE)
            else:
                axs[index].get_legend().remove()
        else:
            sns.lineplot(ax=axs[index], x="Slice (I->S)", y=metric, data=df, errorbar='sd', hue=hue, linewidth=2)

        axs[index].set_ylim(METRICS_TO_YLIM[metric][0], METRICS_TO_YLIM[metric][1])
        ymin, ymax = axs[index].get_ylim()

        # Add labels
        axs[index].set_ylabel(METRIC_TO_AXIS[metric], fontsize=LABELS_FONT_SIZE)
        axs[index].set_xlabel('Axial Slice #', fontsize=LABELS_FONT_SIZE)
        # Increase xticks and yticks font size
        axs[index].tick_params(axis='both', which='major', labelsize=TICKS_FONT_SIZE)

        # Remove spines
        axs[index].spines['right'].set_visible(False)
        axs[index].spines['left'].set_visible(False)
        axs[index].spines['top'].set_visible(False)
        axs[index].spines['bottom'].set_visible(True)

        # Get indices of slices corresponding vertebral levels
        vert, ind_vert, ind_vert_mid = get_vert_indices(df)
        # Insert a vertical line for each intervertebral disc
        for idx, x in enumerate(ind_vert[:-1]):
            axs[index].axvline(df.loc[x, 'Slice (I->S)'], color='black', linestyle='--', alpha=0.5, zorder=0)
        # Insert a text label for each vertebral level
        for idx, x in enumerate(ind_vert_mid, 0):
            # Deal with T1 label (C8 -> T1)
            if vert[x] > 7:
                level = 'T' + str(int(vert[x]) - 7)
                axs[index].text(df.loc[ind_vert_mid[idx], 'Slice (I->S)'], ymin, level, horizontalalignment='center',
                                verticalalignment='bottom', color='black', fontsize=TICKS_FONT_SIZE)
            else:
                level = 'C' + str(int(vert[x]))
                axs[index].text(df.loc[ind_vert_mid[idx], 'Slice (I->S)'], ymin, level, horizontalalignment='center',
                                verticalalignment='bottom', color='black', fontsize=TICKS_FONT_SIZE)

        # Invert x-axis
        axs[index].invert_xaxis()
        # Add only horizontal grid lines
        axs[index].yaxis.grid(True)
        # Move grid to background (i.e. behind other elements)
        axs[index].set_axisbelow(True)

    # Save figure
    if hue:
        filename = 'lineplot_per' + hue + '.png'
    else:
        filename = 'lineplot.png'
    plt.savefig(filename, dpi=500, bbox_inches='tight')
    logger.info('Figure saved: ' + filename)


def r_pvalues(df):
    cols = pd.DataFrame(columns=df.columns)
    p = cols.transpose().join(cols, how='outer')
    for r in df.columns:
        for c in df.columns:
            tmp = df[df[r].notnull() & df[c].notnull()]
            p[r][c] = round(stats.spearmanr(tmp[r], tmp[c])[1], 4)
    return p


def main():

    args = get_parser().parse_args()
    # Get input argments
    input_folder = os.path.abspath(args.i_folder)


    output_folder = args.o_folder
    # Create output folder if does not exist.
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    os.chdir(output_folder)
    # Dump log file there
    if os.path.exists(FNAME_LOG):
        os.remove(FNAME_LOG)
    fh = logging.FileHandler(os.path.join(FNAME_LOG))
    logging.root.addHandler(fh)


    # Analyse T2w perslice
    #################################################################
    logger.info('\nAnalysing T2w CSA perslice in PAM50 anatomical dimension')
    filename = os.path.join(input_folder, "t2w_shape_PAM50.csv")
    df_t2_pam50 = read_t2w_pam50(input_folder)
    #get_number_subjects(df_t2_pam50, session)

    # Keep only VertLevel from C2 to T1
    df_t2_pam50 = df_t2_pam50[df_t2_pam50['VertLevel'] <= 8]
    df_t2_pam50 = df_t2_pam50[df_t2_pam50['VertLevel'] > 1]
    create_lineplot(df_t2_pam50, 'group')
    #compare_metrics_across_group(df_t2_pam50)
    # Aggregate metrics at disc levels:
    # TODO: do for all metrics
    #df_t2w_pam50_avg = avg_metrics(df_t2_pam50)
    #print(df_t2w_pam50_avg)
    #df_t2w_pam50_avg = avg_metrics(df_t2_pam50, metric='MEAN(area)')
    #logger.info('\n Comparing CSA at disc level...')
    #compare_metrics_across_group(df_t2w_pam50_avg, perlevel=True)


if __name__ == "__main__":
    main()