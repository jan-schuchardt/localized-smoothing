import numpy as np
import time
from cvxpy import *
from typing import Tuple, List
from localized_smoothing.graph.utils import p_lower_from_votes, binary_certificate_grid
from torch.jit import Error
from localized_smoothing.graph.smooth import SmoothDiscrete
from localized_smoothing.graph.cluster import Cluster


def base_cert(votes, pre_votes, conf_alpha, pf_plus_att, pf_minus_att,
              n_samples_eval):
    # compute the lower bound on the probability of the majority class
    p_lower = p_lower_from_votes(votes=votes,
                                 pre_votes=pre_votes,
                                 alpha=conf_alpha,
                                 n_samples=n_samples_eval)

    # compute the binary-class certificate 2D grid (for all pairs of ra and rd radii)
    # where grid_binary_class > 0.5 means the instance is robust
    grid_binary_class, *_ = binary_certificate_grid(pf_plus=pf_plus_att,
                                                    pf_minus=pf_minus_att,
                                                    p_emps=p_lower,
                                                    reverse=False,
                                                    progress_bar=True)
    return grid_binary_class


def certify(model: SmoothDiscrete,
            budgets_a: np.ndarray,
            budgets_d: np.ndarray,
            eval_idx: np.ndarray,
            targets: np.ndarray,
            batch_size: int = 10,
            is_mip: bool = False):
    pre_eval_config = model.sample_config.copy()
    pre_eval_config['n_samples'] = model.sample_config.get('n_samples_pre_eval')

    pre_preds, pre_scores = model.sample_noise(batch_size,
                                               sample_config=pre_eval_config)
    predictions, scores = model.sample_noise(batch_size,
                                             sample_config=model.sample_config)
    pre_preds = pre_preds.argmax(1).cpu().numpy()
    predictions = predictions.argmax(1).cpu().numpy()
    scores = scores[:, range(len(pre_preds)),
                    pre_preds]  # only take pre-voted class

    mean_bound, var_bound = model.estimate_bounds(scores)
    ca_map, cd_map = calc_norm_consts(model.sample_config.get('pis'),
                                      model.clustering.rank_matrix)

    n_nodes = len(predictions)
    naive_problem, naive_params, _ = construct_naive_problem(mean=mean_bound,
                                                             variance=var_bound,
                                                             c_a=ca_map,
                                                             c_d=cd_map,
                                                             n_nodes=n_nodes,
                                                             is_mip=is_mip)
    collective_problem, collective_params, _ = construct_collective_problem(
        mean=mean_bound,
        variance=var_bound,
        c_a=ca_map,
        c_d=cd_map,
        clustering=model.clustering,
        n_nodes=n_nodes,
        is_mip=is_mip)
    # get masks
    evaluating = mask_from_indices(eval_idx, n_nodes)
    correct_preds_mask = predictions == targets

    abstain_mask = generate_abstain_mask(mean_bound, var_bound, ca_map, cd_map,
                                         n_nodes, evaluating)
    naive_results = eval_problems(budgets_a, budgets_d, naive_problem,
                                  naive_params, evaluating, abstain_mask,
                                  correct_preds_mask)
    collective_results = eval_problems(budgets_a, budgets_d, collective_problem,
                                       collective_params, evaluating,
                                       abstain_mask, correct_preds_mask)

    return naive_results, collective_results, sum(abstain_mask)


def mask_from_indices(indices, num_vals):
    mask = np.zeros(num_vals)
    mask[indices] = np.ones(len(indices))
    mask = mask.astype('bool')
    return mask


def eval_problems(budgets_a, budgets_d, problem, params, evaluation_mask,
                  abstain_mask, correct_predictions):
    cert_dict = {}
    for cert_type in ['ratio', 'accuracy']:
        prediction_mask = generate_prediction_mask(cert_type,
                                                   correct_predictions,
                                                   evaluation_mask,
                                                   abstain_mask)
        num_nodes = evaluation_mask.sum()
        results = eval_budgets(budgets_a, budgets_d, problem, params,
                               prediction_mask, num_nodes)
        cert_dict[cert_type] = results
    return cert_dict


def eval_budgets(budgets_a, budgets_d, problem, params, prediction_mask,
                 num_nodes):
    cert_dict = {}
    for ba in budgets_a:
        for bd in budgets_d:
            params['budget_a'].value = ba
            params['budget_d'].value = bd
            params['prediction_mask'].value = prediction_mask.astype('int')
            try:
                start = time.time()
                n_certified = prediction_mask.sum() - problem.solve()
                end = time.time()

            except SolverError:
                n_certified = -1
            cert_dict[(ba, bd)] = n_certified / num_nodes
    return cert_dict


def generate_abstain_mask(mean_bound, var_bound, ca_map, cd_map, n_nodes,
                          evaluating):
    # should be MIP here to get the mask
    naive_problem, naive_params, attacked = construct_naive_problem(
        mean=mean_bound,
        variance=var_bound,
        c_a=ca_map,
        c_d=cd_map,
        n_nodes=n_nodes,
        is_mip=True)
    naive_params['budget_a'].value = 0
    naive_params['budget_d'].value = 0
    naive_params['prediction_mask'].value = evaluating.astype('int')
    _ = evaluating.sum() - naive_problem.solve()
    
    return attacked.value.astype('bool')


def generate_prediction_mask(cert_type, correct_predictions, evaluating,
                             abstaining):
    assert cert_type in ['ratio', 'accuracy'
                        ], "wrong cert-type. Can only be 'ratio' or 'accuracy'"
    if cert_type == 'ratio':
        mask = (evaluating & ~abstaining)
    elif cert_type == 'accuracy':
        mask = (correct_predictions & evaluating & ~abstaining)
    return mask


def group_by_cluster(data, clustering: Cluster):
    return [
        data[clustering.get_cluster_idx(i)]
        for i in range(clustering.n_clusters)
    ]


def construct_naive_problem(mean, variance, c_a, c_d, n_nodes, is_mip):
    budget_a = Parameter(name='budget_a')
    budget_d = Parameter(name='budget_a')
    prediction_mask = Parameter(shape=(n_nodes), name='prediction_mask')
    r = np.max(c_a) * budget_a + np.max(c_d) * budget_d
    r_max = np.log(1 + (1 / variance)**2 * (mean - 1 / 2)**2)
    attacked = Variable(shape=(n_nodes), boolean=is_mip)
    constraints = [r >= multiply(attacked, r_max)]
    if not is_mip:
        constraints.extend([0 <= attacked, attacked <= 1])
    objective = sum(multiply(attacked, prediction_mask))
    problem = Problem(Maximize(objective), constraints=constraints)

    params = {
        'budget_a': budget_a,
        'budget_d': budget_d,
        'prediction_mask': prediction_mask
    }
    return problem, params, attacked


def construct_collective_problem(mean, variance, c_a, c_d, clustering: Cluster,
                                 n_nodes, is_mip):
    budget_a = Parameter(name='budget_a')
    budget_d = Parameter(name='budget_d')
    prediction_mask = Parameter(shape=(n_nodes), name='prediction_mask')
    nc = clustering.n_clusters
    r_a = Variable(shape=(nc), name='r_a')
    r_d = Variable(shape=(nc), name='r_d')
    attacked = Variable(shape=(n_nodes), name='attacked', boolean=is_mip)
    constraints = []
    r_max = np.log(1 + (1 / variance)**2 * (mean - 1 / 2)**2)
    for c in range(nc):
        r = r_a @ c_a[c, :] + r_d @ c_d[c, :]
        c_mask = clustering.get_cluster_idx(c).bool().numpy()
        constraints.append(r >= multiply(attacked[c_mask], r_max[c_mask]))
    if not is_mip:
        constraints.extend([0 <= attacked, attacked <= 1])

    constraints.extend([0 <= r_a, 0 <= r_d])
    constraints.append(sum(r_a) <= budget_a)
    constraints.append(sum(r_d) <= budget_d)
    objective = sum(multiply(attacked, prediction_mask))
    problem = Problem(Maximize(objective), constraints=constraints)

    params = {
        'budget_a': budget_a,
        'budget_d': budget_d,
        'prediction_mask': prediction_mask
    }
    return problem, params, attacked


def calc_norm_consts(sample_probs: List[Tuple[float, float]],
                     rank_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Creating the normalizing vectors c_a and c_d for the additions and deletions

    Parameters
    ----------
    sample_probs : List[Tuple[float, float]]
        Sampling probabilities for zeroes and ones. Increasing order
    ranking : np.ndarray [n_cluster, n_cluster]
        matrix with the ranking from cluster i to j based on the affinity
    
    Returns
    ----------
    Tuple[np.ndarray, np.ndarray]
        vectors c_a and c_d respectively
    """
    ca, cd = [], []
    for tp, tm in sample_probs:
        ca.append(((1 - tm)**2) / tp + tm**2 / (1 - tp))
        cd.append((tp**2) / (1 - tm) + (1 - tp)**2 / tm)
    ca = np.array(ca)
    cd = np.array(cd)
    return np.log(ca[rank_matrix]), np.log(cd[rank_matrix])
