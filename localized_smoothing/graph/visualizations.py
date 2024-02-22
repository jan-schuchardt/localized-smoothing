import torch
import matplotlib.pyplot as plt
from localized_smoothing.graph.utils import load_obj, load_and_standardize
from os.path import join
import numpy as np


def extract_results_rd(results, ra, maxrd):
    res_list = []
    for rd in range(maxrd):
        res_list.append(results[(ra, rd)])
    return res_list


def extract_results_ra(results, rd, maxra):
    res_list = []
    for ra in range(maxra):
        res_list.append(results[(ra, rd)])
    return res_list


def plot_baseline(pert_type,
                  result_type,
                  test,
                  model='both',
                  drawstyle='steps-post'):
    #base_1 = torch.load('/nfs/homedirs/wollschl/staff/sparse_smoothing/models/cert-baseline-stand_1')
    #base_2 = torch.load('/nfs/homedirs/wollschl/staff/sparse_smoothing/models/cert-baseline-stand_2')
    '''
    base_1 = torch.load('/nfs/homedirs/wollschl/staff/sparse_smoothing/models/cert-baseline-stand_5') #new gcn
    base_2 = torch.load('/nfs/homedirs/wollschl/staff/sparse_smoothing/models/cert-baseline-stand_6')
    base_3 = torch.load('/nfs/homedirs/wollschl/staff/sparse_smoothing/models/cert-baseline-stand_3')
    base_4 = torch.load('/nfs/homedirs/wollschl/staff/sparse_smoothing/models/cert-baseline-stand_4')

    # 100k baseline
    base_1 = torch.load('/nfs/homedirs/wollschl/staff/sparse_smoothing/models/cert-baseline-100k_1')
    base_2 = torch.load('/nfs/homedirs/wollschl/staff/sparse_smoothing/models/cert-baseline-100k_2')
    base_3 = torch.load('/nfs/homedirs/wollschl/staff/sparse_smoothing/models/cert-baseline-100k_3')
    base_4 = torch.load('/nfs/homedirs/wollschl/staff/sparse_smoothing/models/cert-baseline-100k_4')
    '''
    base_3 = torch.load(
        '/nfs/homedirs/wollschl/staff/sparse_smoothing/models/our_models/APPNP/pp_min_0.01_pm_min_0.6_used_with_0.01_0.6'
    )  #new gcn
    base_1 = torch.load(
        '/nfs/homedirs/wollschl/staff/sparse_smoothing/models/our_models/GCNLarge/pp_min_0.01_pm_min_0.6_used_with_0.01_0.6'
    )

    graph = load_and_standardize(
        '/nfs/homedirs/wollschl/staff/localized_smoothing/data/graphs/cora_ml.npz'
    )
    if pert_type in ['deletion', 'd']:
        if model == 'gcn' or model == 'both':
            plot_single_base_d(base_1,
                               'base_gcn_1_d',
                               result_type,
                               test,
                               graph.labels,
                               drawstyle,
                               legend=True)
            #plot_single_base_d(base_2, 'base_gcn_2_d', result_type, test, graph.labels, drawstyle, legend=True)
        if model == 'appnp' or model == 'both':
            plot_single_base_d(base_3,
                               'base_appnp_3_d',
                               result_type,
                               test,
                               graph.labels,
                               drawstyle,
                               legend=True)
            #plot_single_base_d(base_4, 'base_appnp_4_d', result_type, test, graph.labels, drawstyle, legend=True)
    elif pert_type in ['addition', 'a']:
        if model == 'gcn' or model == 'both':
            plot_single_base_a(base_1,
                               'base_gcn_1_a',
                               result_type,
                               test,
                               graph.labels,
                               drawstyle,
                               legend=True)
            #plot_single_base_a(base_2, 'base_gcn_2_a', result_type, test, graph.labels, drawstyle, legend=True)
        if model == 'appnp' or model == 'both':
            plot_single_base_a(base_3,
                               'base_appnp_3_a',
                               result_type,
                               test,
                               graph.labels,
                               drawstyle,
                               legend=True)
            #plot_single_base_a(base_4, 'base_appnp_4_a', result_type, test, graph.labels, drawstyle, legend=True)
    elif pert_type in ['both']:
        if model == 'gcn' or model == 'both':
            plot_single_base_d(base_1,
                               'base_gcn_1_d',
                               result_type,
                               test,
                               graph.labels,
                               drawstyle,
                               legend=True)
            #plot_single_base_d(base_2, 'base_gcn_2_d', result_type, test, graph.labels, drawstyle, legend=True)
            plot_single_base_a(base_1,
                               'base_gcn_1_a',
                               result_type,
                               test,
                               graph.labels,
                               drawstyle,
                               legend=True)
            #plot_single_base_a(base_2, 'base_gcn_2_a', result_type, test, graph.labels, drawstyle, legend=True)
        if model == 'appnp' or model == 'both':
            plot_single_base_d(base_3,
                               'base_appnp_3_d',
                               result_type,
                               test,
                               graph.labels,
                               drawstyle,
                               legend=True)
            #plot_single_base_d(base_4, 'base_appnp_4_d', result_type, test, graph.labels, drawstyle, legend=True)
            plot_single_base_a(base_3,
                               'base_appnp_3_a',
                               result_type,
                               test,
                               graph.labels,
                               drawstyle,
                               legend=True)
            #plot_single_base_a(base_4, 'base_appnp_4_a', result_type, test, graph.labels, drawstyle, legend=True)


def area_under_curve(x, y, pre=True):
    if pre:
        return np.diff(x) @ y[1:]
    else:
        return np.diff(x) @ y[:-1]


def plot_single_base_d(base_dict,
                       label,
                       result_type,
                       test,
                       labels,
                       drawstyle,
                       legend=False):
    correct_pred = labels == base_dict['pre_votes'].argmax(1)
    if test:
        idx = base_dict['idx_test']
    else:
        idx = base_dict['idx_val']
    mask = [True if i in idx else False for i in range(len(correct_pred))]

    if result_type == 'ratio':
        plt.plot((base_dict['grid_base'][mask, 0, :] > 0.5).mean(0),
                 drawstyle=drawstyle,
                 label=label,
                 clip_on=False,
                 zorder=3)
    elif result_type == 'accuracy':
        plt.plot(
            (base_dict['grid_base'][(correct_pred & mask), 0, :] > 0.5).sum(0) /
            sum(mask),
            drawstyle=drawstyle,
            label='Iso. Naïve',
            clip_on=False,
            zorder=3,
            color='C1')
        print(
            'AUC base deletion: ',
            np.sum((base_dict['grid_base'][(correct_pred & mask), 0, :]
                    > 0.5).sum(0) / sum(mask)))
    elif result_type == 'both':
        plt.plot((base_dict['grid_base'][mask, 0, :] > 0.5).mean(0),
                 drawstyle=drawstyle,
                 label=label,
                 clip_on=False,
                 zorder=3)
        plt.plot(
            (base_dict['grid_base'][(correct_pred & mask), 0, :] > 0.5).sum(0) /
            sum(mask),
            drawstyle=drawstyle,
            linestyle=':',
            label=label,
            clip_on=False,
            zorder=3)
    else:
        raise NotImplementedError
    if legend:
        plt.legend()


def plot_single_base_a(base_dict,
                       label,
                       result_type,
                       test,
                       labels,
                       drawstyle,
                       legend=False):
    correct_pred = labels == base_dict['pre_votes'].argmax(1)
    if test:
        idx = base_dict['idx_test']
    else:
        idx = base_dict['idx_val']
    mask = [True if i in idx else False for i in range(len(correct_pred))]

    if result_type == 'ratio':
        plt.plot((base_dict['grid_base'][mask, :, 0] > 0.5).mean(0),
                 drawstyle=drawstyle,
                 label=label,
                 clip_on=False,
                 zorder=3)
    elif result_type == 'accuracy':
        plt.plot(
            (base_dict['grid_base'][(correct_pred & mask), :, 0] > 0.5).sum(0) /
            sum(mask),
            drawstyle=drawstyle,
            label='Iso. Naïve',
            clip_on=False,
            zorder=3,
            color='C0')
        print(
            'AUC base addition: ',
            np.sum(base_dict['grid_base'][(correct_pred & mask), :,
                                          0] > 0.5).sum(0) / sum(mask))
    elif result_type == 'both':
        plt.plot((base_dict['grid_base'][mask, :, 0] > 0.5).mean(0),
                 drawstyle=drawstyle,
                 label=label,
                 clip_on=False,
                 zorder=3)
        plt.plot(
            (base_dict['grid_base'][(correct_pred & mask), :, 0] > 0.5).sum(0) /
            sum(mask),
            drawstyle=drawstyle,
            linestyle=':',
            label=label,
            clip_on=False,
            zorder=3)
    else:
        raise NotImplementedError
    if legend:
        plt.legend()


def plot_ratio_a_collective(df,
                            drawstyle='default',
                            show_baseline=True,
                            title='certified ratio over additions',
                            base_drawstyle='steps-post',
                            base_model='both',
                            maxr=30):
    #fig = plt.figure(figsize=(12, 12))
    plt.title(title)
    for idx in df.index:
        path = df.loc[idx, 'result.result_path']
        min_m = df.loc[idx, 'config.min_m']
        min_p = df.loc[idx, 'config.min_p']
        max_m = df.loc[idx, 'config.max_m']
        max_p = df.loc[idx, 'config.max_p']
        res = load_obj(join('../', path), 'collective_results')
        ratio = res['ratio']
        radii_a = extract_results_ra(ratio, 0, maxr)
        plt.plot(radii_a,
                 label=f'ra_{min_m}_{min_p}_{max_m}_{max_p}',
                 drawstyle=drawstyle,
                 clip_on=False,
                 zorder=3)
    if show_baseline:
        plot_baseline('addition',
                      'ratio',
                      False,
                      drawstyle=base_drawstyle,
                      model=base_model)
    plt.ylim(0, 1)
    #plt.legend()
    #plt.show()


def plot_ratio_d_collective(df,
                            drawstyle='default',
                            show_baseline=True,
                            title='certified ratio over deletions',
                            base_drawstyle='steps-post',
                            base_model='both',
                            maxr=30,
                            type='accuracy'):
    #fig = plt.figure(figsize=(16, 9))
    #plt.title(title)
    for idx in df.index:
        path = df.loc[idx, 'result.result_path']
        min_m = df.loc[idx, 'config.min_m']
        min_p = df.loc[idx, 'config.min_p']
        max_m = df.loc[idx, 'config.max_m']
        max_p = df.loc[idx, 'config.max_p']
        res = load_obj(join('../', path), 'collective_results')
        ratio = res['ratio']
        radii_d = extract_results_rd(ratio, 0, maxr)
        plt.plot(radii_d,
                 label=f'rd_{min_m}_{min_p}_{max_m}_{max_p}',
                 drawstyle=drawstyle,
                 clip_on=False,
                 zorder=3)
    if show_baseline:
        plot_baseline('deletion',
                      'ratio',
                      False,
                      drawstyle=base_drawstyle,
                      model=base_model)
    plt.ylim(0, 1)
    #plt.legend()
    #plt.show()


def plot_acc_both_collective(df,
                             drawstyle='default',
                             show_baseline=True,
                             base_drawstyle='steps-post',
                             base_model='both',
                             maxr=30):
    plot_acc_a_collective(df,
                          drawstyle,
                          show_baseline,
                          base_drawstyle=base_drawstyle,
                          base_model=base_model,
                          maxr=maxr)
    plot_acc_d_collective(df,
                          drawstyle,
                          show_baseline,
                          base_drawstyle=base_drawstyle,
                          base_model=base_model,
                          maxr=maxr)


def plot_acc_a_collective(df,
                          drawstyle='default',
                          show_baseline=True,
                          title='certified accuracy over additions',
                          base_drawstyle='steps-post',
                          base_model='both',
                          maxr=30,
                          type='accuracy'):
    fig = plt.figure(figsize=(16, 9))
    #plt.title(title)
    for idx in df.index:
        path = df.loc[idx, 'result.result_path']
        min_m = df.loc[idx, 'config.min_m']
        min_p = df.loc[idx, 'config.min_p']
        max_m = df.loc[idx, 'config.max_m']
        max_p = df.loc[idx, 'config.max_p']
        res = load_obj(join('../', path), 'collective_results')
        acc = res['accuracy']
        radii_a = extract_results_ra(acc, 0, maxr)
        print('AUC collective: ', np.array(radii_a).sum())
        #plt.plot(radii_a, label=f'Var. cert., add.',drawstyle=drawstyle, color='C1', clip_on=False, zorder=3, linestyle=':')
        #plt.plot(radii_a, label=f'LP localized',drawstyle=drawstyle, color='C1', clip_on=False, zorder=3)
        n = df.loc[idx, 'config.n_clusters']
        plt.plot(radii_a,
                 label=f'collective-{min_m}-{min_p}-{max_m}-{max_p}-{n}',
                 drawstyle=drawstyle,
                 linestyle=':')  #, color='C1')
    if show_baseline:
        plot_baseline('addition',
                      'accuracy',
                      False,
                      drawstyle=base_drawstyle,
                      model=base_model)
    #plt.xlabel('Number of adversarial perturbations')
    plt.xlabel('Number of adversarial additions')
    plt.ylabel('Certified ' + type)
    plt.ylim(0, 1)
    plt.legend()
    #plt.show()


def plot_acc_d_collective(df,
                          drawstyle='default',
                          show_baseline=True,
                          title='certified accuracy over deleteions',
                          base_drawstyle='steps-post',
                          base_model='both',
                          maxr=30,
                          type='accuracy'):
    #fig = plt.figure(figsize=(16, 9))
    #plt.title(title)
    for idx in df.index:
        path = df.loc[idx, 'result.result_path']
        min_m = df.loc[idx, 'config.min_m']
        min_p = df.loc[idx, 'config.min_p']
        max_m = df.loc[idx, 'config.max_m']
        max_p = df.loc[idx, 'config.max_p']
        res = load_obj(join('../', path), 'collective_results')
        acc = res[type]
        radii_d = extract_results_rd(acc, 0, maxr)
        print('AUC collective: ', np.array(radii_d).sum())
        #plt.plot(radii_d, label=f'Var. cert., del.', drawstyle=drawstyle, color='C1', clip_on=False, zorder=3)
        plt.plot(radii_d,
                 label=f'Localized LP',
                 drawstyle=drawstyle,
                 color='C0',
                 clip_on=False,
                 zorder=3)
        n = df.loc[idx, 'config.n_clusters']
        #plt.plot(radii_d, label=f'collective-{min_m}-{min_p}-{max_m}-{max_p}-{n}',
        #         drawstyle=drawstyle)#, color='C1')
    if show_baseline:
        plot_baseline('deletion',
                      'accuracy',
                      False,
                      drawstyle=base_drawstyle,
                      model=base_model)
    plt.ylim(0, 1)
    plt.xlabel('Number of adversarial deletions')
    plt.ylabel('Certified ' + type)
    plt.legend()
    #plt.show()


def plot_acc_a_naive(df,
                     drawstyle='default',
                     show_baseline=True,
                     title='certified accuracy over deleteions',
                     base_drawstyle='steps-post',
                     base_model='both',
                     maxr=30,
                     type='accuracy'):
    #fig = plt.figure(figsize=(16, 9))
    #plt.title(title)
    for idx in df.index:
        path = df.loc[idx, 'result.result_path']
        min_m = df.loc[idx, 'config.min_m']
        min_p = df.loc[idx, 'config.min_p']
        max_m = df.loc[idx, 'config.max_m']
        max_p = df.loc[idx, 'config.max_p']
        res = load_obj(join('../', path), 'naive_results')
        acc = res[type]
        radii_a = extract_results_ra(acc, 0, maxr)
        print('AUC naive: ', np.array(radii_a).sum())

        plt.plot(radii_a,
                 label=f'naive-{min_m}-{min_p}-{max_m}-{max_p}',
                 drawstyle=drawstyle,
                 linestyle=':',
                 color='C1')
        #plt.plot(radii_a, drawstyle=drawstyle, linestyle=':', label='Naïve localized', color='C1', clip_on=False, zorder=3)
    if show_baseline:
        plot_baseline('addition',
                      'accuracy',
                      False,
                      drawstyle=base_drawstyle,
                      model=base_model)
    #plt.ylim(0, 1)
    #plt.legend()
    #plt.show()


def plot_acc_d_naive(df,
                     drawstyle='default',
                     show_baseline=True,
                     title='certified accuracy over deleteions',
                     base_drawstyle='steps-post',
                     base_model='both',
                     maxr=30,
                     type='accuracy'):
    #fig = plt.figure(figsize=(16, 9))
    #plt.title(title)
    for idx in df.index:
        path = df.loc[idx, 'result.result_path']
        min_m = df.loc[idx, 'config.min_m']
        min_p = df.loc[idx, 'config.min_p']
        max_m = df.loc[idx, 'config.max_m']
        max_p = df.loc[idx, 'config.max_p']
        res = load_obj(join('../', path), 'naive_results')
        acc = res[type]
        radii_d = extract_results_rd(acc, 0, maxr)
        print('AUC naive: ', np.array(radii_d).sum())
        #plt.plot(radii_d, label=f'naive-{min_m}-{min_p}-{max_m}-{max_p}', drawstyle=drawstyle, linestyle=':', color='C1')
        plt.plot(radii_d,
                 drawstyle=drawstyle,
                 linestyle=':',
                 color='C0',
                 label='Localized Naïve',
                 clip_on=False,
                 zorder=3)
    if show_baseline:
        plot_baseline('deletion',
                      'accuracy',
                      False,
                      drawstyle=base_drawstyle,
                      model=base_model)
    #plt.ylim(0, 1)
    #plt.legend()
    #plt.show()


def plot_ratio_a(df, maxr=30):
    fig = plt.figure(figsize=(10, 8))
    plt.title('certified ratio over additions')
    for idx in df.index:
        path = df.loc[idx, 'result.result_path']
        min_m = df.loc[idx, 'config.min_m']
        min_p = df.loc[idx, 'config.min_p']
        res = load_obj(join('../', path), 'naive_results')
        ratio = res['ratio']
        radii_a = extract_results_ra(ratio, 0, maxr)
        plt.plot(radii_a, label=f'naive_ra_{min_m}_{min_p}')

        res = load_obj(join('../', path), 'collective_results')
        ratio = res['ratio']
        radii_a = extract_results_ra(ratio, 0, maxr)
        plt.plot(radii_a, label=f'collective_ra_{min_m}_{min_p}')
    plt.legend()
    plt.show()


def plot_ratio_d(df, maxr=30):
    fig = plt.figure(figsize=(10, 8))
    plt.title('certified ratio over deletions')
    for idx in df.index:
        path = df.loc[idx, 'result.result_path']
        min_m = df.loc[idx, 'config.min_m']
        min_p = df.loc[idx, 'config.min_p']
        res = load_obj(join('../', path), 'naive_results')
        ratio = res['ratio']
        radii_d = extract_results_rd(ratio, 0, maxr)
        plt.plot(radii_d, label=f'naive_ra_{min_m}_{min_p}')

        res = load_obj(join('../', path), 'collective_results')
        ratio = res['ratio']
        radii_d = extract_results_rd(ratio, 0, maxr)
        plt.plot(radii_d, label=f'collective_ra_{min_m}_{min_p}')
    plt.legend()
    plt.show()


def plot_acc(df, maxr=30):
    fig = plt.figure(figsize=(8, 6))
    plt.title('certified accuracy over r_a')
    for idx in df.index:
        path = df.loc[idx, 'result.result_path']
        min_m = df.loc[idx, 'config.min_m']
        min_p = df.loc[idx, 'config.min_p']
        res = load_obj(join('../', path), 'naive_results')
        ratio = res['accuracy']
        radii_a = extract_results_ra(ratio, 0, maxr)
        radii_d = extract_results_rd(ratio, 0, maxr)
        plt.plot(radii_a, label=f'naive_ra_{min_m}_{min_p}')

        res = load_obj(join('../', path), 'collective_results')
        radii_a = extract_results_ra(ratio, 0, maxr)
        radii_d = extract_results_rd(ratio, 0, maxr)
        plt.plot(radii_a, label=f'collective_ra_{min_m}_{min_p}')
    plt.legend()
    plt.show()
