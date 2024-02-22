import torch
import numpy as np
from typing import List
from torch_sparse import coalesce
from scipy.sparse import dok_matrix
import matplotlib.pyplot as plt
from localized_smoothing.graph.sparsegraph import SparseGraph
import matplotlib.colors as mcolors
from os.path import join
import pickle
import os
from statsmodels.stats.proportion import proportion_confint
import gmpy2
from tqdm import tqdm


def regions_binary(ra, rd, pf_plus, pf_minus, precision=1000):
    """
    Construct (px, px_tilde, px/px_tilde) regions used to find the certified radius for binary data.

    Intuitively, pf_minus controls rd and pf_plus controls ra.

    Parameters
    ----------
    ra: int
        Number of ones y has added to x
    rd : int
        Number of ones y has deleted from x
    pf_plus : float, 0 <= p_plus <= 1
        The probability to flip a zero to a one
    pf_minus: float, 0 <= p_plus <= 1
        The probability to flip a one to a zero
    precision: int
        Numerical precision for floating point calculations

    Returns
    -------
    regions: array-like, [None, 3]
        Regions of constant probability under px and px_tilde,
    """

    pf_plus, pf_minus = gmpy2.mpfr(pf_plus), gmpy2.mpfr(pf_minus)
    with gmpy2.context(precision=precision):
        if pf_plus == 0:
            px = pf_minus**rd
            px_tilde = pf_minus**ra

            return np.array([[1 - px, 0, float('inf')],
                             [px, px_tilde, px / px_tilde],
                             [0, 1 - px_tilde, 0]])

        if pf_minus == 0:
            px = pf_plus**ra
            px_tilde = pf_plus**rd
            return np.array([
                [1 - px, 0, float('inf')],
                [px, px_tilde, px / px_tilde],
                [0, 1 - px_tilde, 0],
            ])
        max_q = ra + rd
        i_vec = np.arange(0, max_q + 1)

        T = ra * ((pf_plus / (1 - pf_plus)) ** i_vec) + \
            rd * ((pf_minus / (1 - pf_minus)) ** i_vec)

        ratio = np.zeros_like(T)
        px = np.zeros_like(T)
        px[0] = 1

        for q in range(0, max_q + 1):
            ratio[q] = (pf_plus/(1-pf_minus)) ** (q - rd) * \
                (pf_minus/(1-pf_plus)) ** (q - ra)

            if q == 0:
                continue

            for i in range(1, q + 1):
                px[q] = px[q] + ((-1)**(i + 1)) * T[i] * px[q - i]
            px[q] = px[q] / q

        scale = ((1 - pf_plus)**ra) * ((1 - pf_minus)**rd)

        px = px * scale

        regions = np.column_stack((px, px / ratio, ratio))
        if pf_plus + pf_minus > 1:
            # reverse the order to maintain decreasing sorting
            regions = regions[::-1]
        return regions


def compute_rho(regions, p_emp, verbose=False, is_sorted=True, reverse=False):
    """
    Compute the worst-case probability of the adversary.
    For the binary-class certificate if rho>0.5 the instance is certifiable robust.

    Parameters
    ----------
    regions : array-like, [?, 3]
        Regions of constant probability under p_x and p_y,
    p_emp : float
        Empirical probability of the majority class
    verbose : bool
        Verbosity
    is_sorted : bool
        Whether the regions are sorted, e.g. regions from `regions_binary` are automatically sorted.
    reverse : bool
        Whether to consider the sorting in reverse order which we need for computing an upper_bound.

    Returns
    -------
    p_adver : float
        The worst-case probability of the adversary.
    """
    if is_sorted:
        sorted_regions = regions
    else:
        # sort in descending order
        sorted_regions = sorted(list(regions), key=lambda a: a[2], reverse=True)
    sorted_regions = reversed(sorted_regions) if reverse else sorted_regions

    if verbose:
        region_sum = sum(map(lambda x: x[0], regions))
        print('region_sum_is', region_sum)

    acc_p_clean = 0.0
    acc_p_adver = 0.0

    for i, (p_clean, p_adver, _) in enumerate(sorted_regions):
        # break early so the sums only reflect up to H*-1
        if acc_p_clean + p_clean >= p_emp:
            break
        if p_clean > 0:
            acc_p_clean += p_clean
            acc_p_adver += p_adver

    rho = acc_p_adver

    if verbose:
        print('clean', float(acc_p_clean), 'adver', float(acc_p_adver),
              'counter={}/{}'.format(i, len(regions)))

    # there is some probability left
    if p_emp - acc_p_clean > 0 and i < len(regions):
        addition = (p_emp - acc_p_clean) * (p_adver / p_clean)
        rho += addition

        if verbose:
            print('ratio', float(p_adver / p_clean), 'diff',
                  float(p_emp - acc_p_clean), 'addition', float(addition))
            print(float(p_adver), float(p_clean))
            print(rho > acc_p_adver)

    return rho


def compute_rho_for_many(regions, p_emps, is_sorted=True, reverse=False):
    """
    Compute the worst-case probability of the adversary for many p_emps at once.

    Parameters
    ----------
    regions : array-like, [?, 3]
        Regions of constant probability under p_x and p_y,
    p_emps : array-like [?]
        Empirical probabilities per node.
    is_sorted : bool
        Whether the regions are sorted, e.g. regions from `regions_binary` are automatically sorted.
    reverse : bool
        Whether to consider the sorting in reverse order.

    Returns
    -------
    p_adver : array-like [?]
        The worst-case probability of the adversary.
    """
    sort_direction = -1 if reverse else 1
    if not is_sorted:
        o = regions[:, 2].argsort()[::-sort_direction]
        regions = regions[o]
    else:
        regions = regions[::sort_direction]

    # add one empty region to have easier indexing
    regions = np.row_stack(([0, 0, 0], regions))

    cumsum = np.cumsum(regions[:, :2], 0)
    h_stars = (cumsum[:, 0][:, None] >= p_emps).argmax(0)
    h_stars[h_stars > 0] -= 1

    h_star_cumsums = cumsum[h_stars]

    acc_p_clean = h_star_cumsums[:, 0]
    acc_p_adver = h_star_cumsums[:, 1]

    # add the missing probability for those that need it
    flt = (p_emps - acc_p_clean > 0) & (h_stars + 1 < len(regions))
    addition = (p_emps[flt] - acc_p_clean[flt]) * \
        regions[h_stars[flt] + 1, 1] / regions[h_stars[flt] + 1, 0]
    acc_p_adver[flt] += addition

    acc_p_adver[h_stars == -1] = 0

    return acc_p_adver.astype('float')


def max_radius_for_p_emp(pf_plus,
                         pf_minus,
                         p_emp,
                         which,
                         upper=100,
                         verbose=False):
    """
    Find the maximum radius we can certify individually (either ra or rd) using bisection.

    Parameters
    ----------
    pf_plus : float, 0 <= p_plus <= 1
        The probability to flip a zero to a one
    pf_minus: float, 0 <= p_plus <= 1
        The probability to flip a one to a zero
    p_emp : float
        Empirical probability of the majority class
    which : string
        'ra': find max_{ra, rd=0}
        'rd': find max_{rd, ra=0}
    upper : int
        An upper bound on the maximum radius
    verbose : bool
        Verbosity.

    Returns
    -------
    max_r : int
        The maximum certified radius s.t. the probability of the adversary is above 0.5.

    """
    initial_upper = upper
    lower = 1
    r = 1

    while lower < upper:
        r = lower + (upper - lower) // 2
        if which == 'ra':
            ra = r
            rd = 0
        elif which == 'rd':
            ra = 0
            rd = r
        else:
            raise ValueError('which can only be "ra" or "rd"')

        cur_rho = compute_rho(
            regions_binary(ra=ra, rd=rd, pf_plus=pf_plus, pf_minus=pf_minus),
            p_emp)
        if verbose:
            print(r, float(cur_rho))

        if cur_rho > 0.5:
            if lower == r:
                break
            lower = r
        else:
            upper = r

    if r == initial_upper or r == initial_upper - 1:
        if verbose:
            print('r = upper, restart the search with a larger upper bound')
        return max_radius_for_p_emp(pf_plus=pf_plus,
                                    pf_minus=pf_minus,
                                    p_emp=p_emp,
                                    which=which,
                                    upper=2 * upper,
                                    verbose=verbose)

    return r


def min_p_emp_for_radius_1(pf_plus, pf_minus, which, lower=0.5, verbose=False):
    """
    Find the smallest p_emp for which we can certify a radius of 1 using bisection.


    Parameters
    ----------
    pf_plus : float, 0 <= p_plus <= 1
        The probability to flip a zero to a one
    pf_minus: float, 0 <= p_plus <= 1
        The probability to flip a one to a zero
    which : string
        'ra': find min_{p_emp, ra=1, rd=0}
        'rd': find min_{p_emp, rd=1, ra=0}
    lower : float
        A lower bound on the minimum p_emp.
    verbose : bool
        Verbosity.

    Returns
    -------
    min_p_emp : float
        The minimum p_emp.
    """
    initial_lower = lower
    upper = 1
    p_emp = 0

    if which == 'ra':
        ra = 1
        rd = 0
    elif which == 'rd':
        ra = 0
        rd = 1
    else:
        raise ValueError('which can only be "ra" or "rd"')

    while lower < upper:
        p_emp = lower + (upper - lower) / 2

        cur_rho = compute_rho(
            regions_binary(ra=ra, rd=rd, pf_plus=pf_plus, pf_minus=pf_minus),
            p_emp)
        if verbose:
            print(p_emp, float(cur_rho))

        if cur_rho < 0.5:
            if lower == p_emp:
                break
            lower = p_emp
        elif abs(cur_rho - 0.5) < 1e-10:
            break
        else:
            upper = p_emp

    if p_emp <= initial_lower:
        if verbose:
            print(
                'p_emp <= initial_lower, restarting the search with a smaller lower bound'
            )
        return min_p_emp_for_radius_1(pf_plus=pf_plus,
                                      pf_minus=pf_minus,
                                      which=which,
                                      lower=lower * 0.5,
                                      verbose=verbose)

    return p_emp


def binary_certificate_grid(pf_plus,
                            pf_minus,
                            p_emps,
                            reverse=False,
                            regions=None,
                            max_ra=None,
                            max_rd=None,
                            progress_bar=True):
    """
    Compute rho for all given p_emps and for all combinations of radii up to the maximum radii.

    Parameters
    ----------
    pf_plus : float, 0 <= p_plus <= 1
        The probability to flip a zero to a one
    pf_minus: float, 0 <= p_plus <= 1
        The probability to flip a one to a zero
    p_emps : array-like [n_nodes]
        Empirical probabilities per node.
    reverse : bool
        Whether to consider the sorting in reverse order.
    regions : dict
        A pre-computed dictionary of regions
    progress_bar : bool
        Whether to show a tqdm progress bar

    Returns
    -------
    radii : array-like, [n_nodes, max_ra, max_rd]
        Probabilities of the adversary. Node is certified if [:, :, :]>0.5
    regions : dict
        A pre-computed dictionary of regions
    max_ra : int
        Maximum certified addition radius
    max_rd : int
        Maximum certified deletion radius
    """
    if progress_bar:

        def bar(loop):
            return tqdm(loop)
    else:

        def bar(loop):
            return loop

    if regions is None:
        # compute the maximum possible ra and rd we can certify for the largest p_emp
        if max_ra is None or max_rd is None:
            max_p_emp = p_emps.max()
            max_ra = max_radius_for_p_emp(pf_plus=pf_plus,
                                          pf_minus=pf_minus,
                                          p_emp=max_p_emp,
                                          which='ra',
                                          upper=100)
            max_rd = max_radius_for_p_emp(pf_plus=pf_plus,
                                          pf_minus=pf_minus,
                                          p_emp=max_p_emp,
                                          which='rd',
                                          upper=100)
            min_p_emp = min(min_p_emp_for_radius_1(pf_plus, pf_minus, 'ra'),
                            min_p_emp_for_radius_1(pf_plus, pf_minus, 'rd'))

            print(
                f'max_ra={max_ra}, max_rd={max_rd}, min_p_emp={min_p_emp:.4f}')

        regions = {}
        for ra in bar(range(max_ra + 2)):
            for rd in range(max_rd + 2):
                regions[(ra, rd)] = regions_binary(ra=ra,
                                                   rd=rd,
                                                   pf_plus=pf_plus,
                                                   pf_minus=pf_minus)

    n_nodes = len(p_emps)
    arng = np.arange(n_nodes)
    radii = np.zeros((n_nodes, max_ra + 2, max_rd + 2))

    for (ra, rd), regions_ra_rd in bar(regions.items()):
        if ra + rd == 0:
            radii[arng, ra, rd] = 1
        else:
            radii[arng, ra, rd] = compute_rho_for_many(regions=regions_ra_rd,
                                                       p_emps=p_emps,
                                                       is_sorted=True,
                                                       reverse=reverse)

    return radii, regions, max_ra, max_rd


def p_lower_from_votes(votes, pre_votes, alpha, n_samples):
    """
    Estimate a lower bound on the probability of the majority class using a Binomial confidence interval.

    Parameters
    ----------
    votes: array_like [n_nodes, n_classes]
        Votes per class for each sample
    pre_votes: array_like [n_nodes, n_classes]
        Votes (based on fewer samples) to determine the majority (and the second best) class
    alpha : float
        Significance level
    n_samples : int
        Number of MC samples
    Returns
    -------
    p_lower: array-like [n_nodes]
        Lower bound on the probability of the majority class

    """
    # Multiple by 2 since we are only need a single side
    n_best = votes[np.arange(votes.shape[0]), pre_votes.argmax(1)]
    p_lower = proportion_confint(n_best,
                                 n_samples,
                                 alpha=2 * alpha,
                                 method="beta")[0]
    return p_lower


def sample_multiple_graphs_attr(attr_idx, pf_plus_att, pf_minus_att, n, d,
                                nsamples):
    """
    Perturb the structure and node attributes.
    Parameters
    ----------
    attr_idx : torch.Tensor [2, ?]
        The indices of the non-zero attributes.
    edge_idx : torch.Tensor [2, ?]
        The indices of the edges.
    pf_plus_att : float
        Sampling probabilities for the bits that are at 0
    pf_minus_att : float
        Sampling probabilities for the bits that are at 1
    n : int
        Number of nodes
    d : int
        Number of features
    nsamples : int
        Number of samples
    Returns
    -------
    attr_idx: torch.Tensor [2, ?]
        The indices of the non-zero attributes after perturbation.
    edge_idx: torch.Tensor [2, ?]
        The indices of the edges after perturbation.
    """

    if pf_minus_att + pf_plus_att > 0:
        per_attr_idx = sparse_perturb_multiple(data_idx=attr_idx,
                                               n=n,
                                               m=d,
                                               undirected=False,
                                               pf_minus=pf_minus_att,
                                               pf_plus=pf_plus_att,
                                               nsamples=nsamples,
                                               offset_both_idx=False)
    else:
        per_attr_idx = copy_idx(idx=attr_idx,
                                dim_size=n,
                                ncopies=nsamples,
                                offset_both_idx=False)

    return per_attr_idx


def sample_multiple_graphs_edges(edge_idx, n, nsamples):
    """
    Parameters
    ----------
    edge_idx : [type]
        [description]
    n : [type]
        [description]
    nsamples : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    return copy_idx(idx=edge_idx,
                    dim_size=n,
                    ncopies=nsamples,
                    offset_both_idx=True)


def sparse_perturb_multiple(data_idx, pf_minus, pf_plus, n, m, undirected,
                            nsamples, offset_both_idx):
    """
    Randomly flip bits.
    Parameters
    ----------
    data_idx: torch.Tensor [2, ?]
        The indices of the non-zero elements
    pf_minus: float, 0 <= p_plus <= 1
        The probability to flip a one to a zero
    pf_plus : float, 0 <= p_plus <= 1
        The probability to flip a zero to a one
    n : int
        The shape of the tensor
    m : int
        The shape of the tensor
    undirected : bool
        If true for every (i, j) also perturb (j, i)
    nsamples : int
        Number of perturbed samples
    offset_both_idx : bool
        Whether to offset both matrix indices (for adjacency matrix)
    Returns
    -------
    perturbed_data_idx: torch.Tensor [2, ?]
        The indices of the non-zero elements of multiple concatenated matrices
        after perturbation
    """
    if undirected:
        # select only one direction of the edges, ignore self loops
        data_idx = data_idx[:, data_idx[0] < data_idx[1]]
    idx_copies = copy_idx(data_idx, n, nsamples, offset_both_idx)
    w_existing = torch.ones_like(idx_copies[0])
    to_del = torch.cuda.BoolTensor(idx_copies.shape[1]).bernoulli_(pf_minus)
    w_existing[to_del] = 0

    if offset_both_idx:
        assert n == m
        nadd_persample_np = np.random.binomial(
            n * m, pf_plus, size=nsamples)  # 6x faster than PyTorch
        nadd_persample = torch.cuda.FloatTensor(nadd_persample_np)
        nadd_persample_with_repl = torch.round(
            torch.log(1 - nadd_persample / (n * m)) / np.log(1 - 1 /
                                                             (n * m))).long()
        nadd_with_repl = nadd_persample_with_repl.sum()
        to_add = data_idx.new_empty([2, nadd_with_repl])
        to_add[0].random_(n * m)
        to_add[1] = to_add[0] % m
        to_add[0] = to_add[0] // m
        to_add = offset_idx(to_add, nadd_persample_with_repl, m, [0, 1])
        if undirected:
            # select only one direction of the edges, ignore self loops
            to_add = to_add[:, to_add[0] < to_add[1]]
    else:
        nadd = np.random.binomial(nsamples * n * m,
                                  pf_plus)  # 6x faster than PyTorch
        nadd_with_repl = int(
            np.round(
                np.log(1 - nadd / (nsamples * n * m)) /
                np.log(1 - 1 / (nsamples * n * m))))
        to_add = data_idx.new_empty([2, nadd_with_repl])
        to_add[0].random_(nsamples * n * m)
        to_add[1] = to_add[0] % m
        to_add[0] = to_add[0] // m

    w_added = torch.ones_like(to_add[0])

    if offset_both_idx:
        mb = nsamples * m
    else:
        mb = m

    # if an edge already exists but has been removed do not add it back
    # hence we coalesce with the min value
    joined, weights = coalesce(torch.cat((idx_copies, to_add), 1),
                               torch.cat((w_existing, w_added), 0),
                               nsamples * n, mb, 'min')

    per_data_idx = joined[:, weights > 0]

    if undirected:
        per_data_idx = torch.cat((per_data_idx, per_data_idx[[1, 0]]), 1)

    # Check that there are no off-diagonal edges
    # if offset_both_idx:
    #     batch0 = to_add[0] // n
    #     batch1 = to_add[1] // n
    #     assert torch.all(batch0 == batch1)

    return per_data_idx


def copy_idx(idx: torch.LongTensor, dim_size: int, ncopies: int,
             offset_both_idx: bool):
    """
    Randomly flip bits.
    Parameters
    ----------
    idx: torch.LongTensor
        something
    dim_size: int 
        size of dimension
    ncopies: int
        number of copies
    offset_both_idx: bool
        some offset
    Returns
    -------
    idx_copies: torch.Tensor [2, ?]
        Copied indices simulating independent graphs due to the offset
    """
    idx_copies = idx.repeat(1, ncopies)

    offset = dim_size * torch.arange(
        ncopies, dtype=torch.long, device=idx.device)[:, None].expand(
            ncopies, idx.shape[1]).flatten()

    if offset_both_idx:
        idx_copies += offset[None, :]
    else:
        idx_copies[0] += offset

    return idx_copies


def offset_idx(idx_mat: torch.LongTensor,
               lens: torch.LongTensor,
               dim_size: int,
               indices: List[int] = [0]):
    offset = dim_size * torch.arange(
        len(lens), dtype=torch.long, device=idx_mat.device).repeat_interleave(
            lens, dim=0)

    idx_mat[indices, :] += offset[None, :]
    return idx_mat


def calc_pis(min_p, max_p, min_m, max_m, n_clusters):
    pi_ps = np.linspace(start=min_p, stop=max_p, num=n_clusters)
    pi_ms = np.linspace(start=min_m, stop=max_m, num=n_clusters)
    return np.array(list(zip(pi_ps, pi_ms)))


def load_and_standardize(file_name) -> SparseGraph:
    """
    Run gust.standardize() + make the attributes binary.

    Parameters
    ----------
    file_name
        Name of the file to load.
    Returns
    -------
    graph: gust.SparseGraph
        The standardized graph

    """
    with np.load(file_name, allow_pickle=True) as loader:
        loader = dict(loader)
        if 'type' in loader:
            del loader['type']
        graph = SparseGraph.from_flat_dict(loader)

    graph.standardize()

    # binarize
    graph._flag_writeable(True)
    graph.adj_matrix[graph.adj_matrix != 0] = 1
    graph.attr_matrix[graph.attr_matrix != 0] = 1
    graph._flag_writeable(False)

    return graph


def init_list_dict(num_inputs):
    data = dict()
    for i in range(num_inputs):
        data[i] = []
    return data


def split(labels, n_per_class=20, seed=0):
    """
    Randomly split the training data.

    Parameters
    ----------
    labels: array-like [n_nodes]
        The class labels
    n_per_class : int
        Number of samples per class
    seed: int
        Seed

    Returns
    -------
    split_train: array-like [n_per_class * nc]
        The indices of the training nodes
    split_val: array-like [n_per_class * nc]
        The indices of the validation nodes
    split_test array-like [n_nodes - 2*n_per_class * nc]
        The indices of the test nodes
    """
    np.random.seed(seed)
    nc = labels.max() + 1

    split_train, split_val = [], []
    for l in range(nc):
        perm = np.random.permutation((labels == l).nonzero()[0])
        split_train.append(perm[:n_per_class])
        split_val.append(perm[n_per_class:2 * n_per_class])

    split_train = np.random.permutation(np.concatenate(split_train))
    split_val = np.random.permutation(np.concatenate(split_val))

    assert split_train.shape[0] == split_val.shape[0] == n_per_class * nc

    split_test = np.setdiff1d(np.arange(len(labels)),
                              np.concatenate((split_train, split_val)))

    return split_train, split_val, split_test


colors_map = np.random.permutation(list(mcolors.CSS4_COLORS.keys()))


def save_obj(obj, dir, name):
    with open(join(dir, name + '.pkl'), 'wb') as f:
        print(os.path.abspath(join(dir, name + '.pkl')))
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(dir, name):
    with open(join(dir, name + '.pkl'), 'rb') as f:
        return pickle.load(f)


def pretty(d, indent=0):
    for key, value in d.items():
        print('\t' * indent + str(key))
        if isinstance(value, dict):
            pretty(value, indent + 1)
        else:
            print('\t' * (indent + 1) + str(value))
