from collections import namedtuple
import numpy as np
from scipy import stats
import gurobipy as gp
from gurobipy import GRB
from sklearn import tree
from src.solver_monitor import (
    SolverMonitor, combined_callback, configure_solver_logging,
    iteration_log_path, solver_record,
)

class softPUOCT:

    def __init__(self, max_depth=3, min_samples_split=2, alpha=0, warmstart=True,
                 timelimit=600, output=False, log_file=None, iteration_callback=None,
                 random_state=42, concise_progress=True, heartbeat_seconds=60,
                 save_gurobi_log=True, progress_context=""):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.alpha = alpha
        self.warmstart = warmstart
        self.timelimit = timelimit
        self.output = output
        self.trained = False
        self.optgap = None
        self.log_file = log_file
        self.iteration_callback = iteration_callback
        self.random_state = random_state
        self.concise_progress = concise_progress
        self.heartbeat_seconds = heartbeat_seconds
        self.save_gurobi_log = save_gurobi_log
        self.progress_context = progress_context
        self.solve_history = []
        self.diff_history = []
        self.converged_ = False
        self.n_iter_ = 0
        self.last_solver_status = None


        self.n_index = [i+1 for i in range(2 ** (self.max_depth + 1) - 1)]
        self.b_index = self.n_index[:-2**self.max_depth] # branch nodes
        self.l_index = self.n_index[-2**self.max_depth:] # leaf nodes

        self._a = None
        self._b = None
        self._c = None
        self._d = None
        self._l = None
        self.leaf_pos_prob = {}
        self.p_u_final = None

    def fit_pu(self, X, pos_mask, unlabeled_mask, max_iter=6, tol=1e-3, init_pos_prob=0.2, verbose=True):

        n = X.shape[0]
        if pos_mask.sum() == 0:
            raise ValueError("No labeled positive samples provided.")
        if not np.all(pos_mask + unlabeled_mask):
            raise ValueError("Every sample must be either positive-labeled or unlabeled in PU setup.")

        self.n, self.p = X.shape
        self.scales = np.max(X, axis=0)
        self.scales[self.scales == 0] = 1
        self.labels = np.array([0,1])

        p = np.zeros(n)
        p[pos_mask] = 1.0
        p[unlabeled_mask] = init_pos_prob

        self.solve_history = []
        self.diff_history = []
        self.converged_ = False
        for it in range(max_iter):
            self._current_iteration = it + 1
            if verbose:
                print(f"[PU] Iteration {it+1}/{max_iter}")

            w_pos = p.copy()
            w_neg = 1.0 - w_pos


            m, a, b, c, d, l, z, M_pos_vars, M_neg_vars = self._buildMIP_weighted(X/self.scales, w_pos, w_neg)

            if self.warmstart:
                pseudo_y = (w_pos >= 0.5).astype(int)
                try:
                    self._setStart(X/self.scales, pseudo_y, a, c, d, l)
                except Exception:
                    pass

            if self.concise_progress:
                print(f"[EM start] iteration={it + 1}/{max_iter} | {self.progress_context}", flush=True)
            monitor = SolverMonitor(
                heartbeat_seconds=self.heartbeat_seconds,
                show_concise_progress=self.concise_progress,
                context=self.progress_context,
                iteration=it + 1,
                max_iterations=max_iter,
                sense=int(m.ModelSense),
            )
            m.optimize(combined_callback(monitor))
            if m.SolCount > 0:
                self.optgap = m.MIPGap
            else:
                self.optgap = None
            if m.SolCount <= 0:
                self.last_solver_status = int(m.Status)
                raise RuntimeError(f"SoftPUOCT iteration {it + 1} returned no feasible solution (status={m.Status})")

            # extract solution variables
            self._a = {ind:a[ind].x for ind in a}
            self._b = {ind:b[ind].x for ind in b}
            self._c = {ind:c[ind].x for ind in c}
            self._d = {ind:d[ind].x for ind in d}
            self._l = {ind:l[ind].x for ind in l}

            z_sol = np.zeros((n, len(self.l_index)), dtype=int)
            lindex_to_pos = {self.l_index[idx]: idx for idx in range(len(self.l_index))}
            for i in range(n):
                for t in self.l_index:
                    if z[(i,t)].X > 0.5:
                        z_sol[i, lindex_to_pos[t]] = 1

            leaf_prob = {}
            for t in self.l_index:
                Mp = M_pos_vars[t].X
                Mn = M_neg_vars[t].X
                denom = Mp + Mn
                if denom <= 0:
                    prob = 0.0
                else:
                    prob = Mp / denom
                leaf_prob[t] = prob
                if verbose:
                    print(f"  Leaf {t}: M_pos={Mp:.4f}, M_neg={Mn:.4f}, pos_prob={prob:.4f}")

            new_p = p.copy()
            for i in range(n):
                for t in self.l_index:
                    if z[(i,t)].X > 0.5:
                        new_p[i] = leaf_prob[t]
                        break

            diff = np.abs(new_p - p).mean()
            record = solver_record(m, iteration=it + 1, diff=float(diff), monitor=monitor)
            self.solve_history.append(record)
            self.diff_history.append(float(diff))
            self.last_solver_status = record["status"]
            self.n_iter_ = it + 1
            if self.iteration_callback is not None:
                self.iteration_callback(record, self)
            if self.concise_progress:
                gap = "NA" if record["mip_gap"] is None else f"{100.0 * record['mip_gap']:.2f}%"
                print(
                    f"[EM done] iteration={it + 1}/{max_iter} | diff={diff:.6f} | "
                    f"converged={'yes' if diff < tol else 'no'} | "
                    f"status={record['status_name']} | gap={gap} | "
                    f"solver_time={record['runtime_seconds']:.2f}s",
                    flush=True,
                )
            if verbose:
                print(f"  mean |p_new - p_old| = {diff:.6f}")
            p = new_p

            self.leaf_pos_prob = leaf_prob

            if diff < tol:
                self.converged_ = True
                if verbose:
                    print("[PU] Converged.")
                break

        self.p_u_final = p
        self.trained = True
        if verbose:
            print("[PU] Training finished. Final unlabeled positive probabilities stored in p_u_final.")
        return self

    def _solver_log_path(self):
        return iteration_log_path(self.log_file, getattr(self, "_current_iteration", None))

    def _buildMIP_weighted(self, x, w_pos, w_neg):

        n_orig, p = x.shape

        unique_x, inv_idx = np.unique(x, axis=0, return_inverse=True)
        n_group = unique_x.shape[0]

        wpos_g = np.zeros(n_group, dtype=float)
        wneg_g = np.zeros(n_group, dtype=float)
        counts = np.zeros(n_group, dtype=float)
        for i, g in enumerate(inv_idx):
            wpos_g[g] += float(w_pos[i])
            wneg_g[g] += float(w_neg[i])
            counts[g] += 1.0

        self.n = n_group
        self.p = p

        self.n_index = [i + 1 for i in range(2 ** (self.max_depth + 1) - 1)]
        self.b_index = self.n_index[:-2 ** self.max_depth]
        self.l_index = self.n_index[-2 ** self.max_depth:]

        m = gp.Model('m_weighted_agg')
        configure_solver_logging(
            m,
            show_raw_console=self.output,
            save_log=self.save_gurobi_log,
            log_file=self._solver_log_path(),
        )
        m.Params.TimeLimit = self.timelimit
        m.Params.Threads = 0
        m.modelSense = GRB.MINIMIZE

        a = m.addVars(p, self.b_index, vtype=GRB.BINARY, name='a')
        b = m.addVars(self.b_index, vtype=GRB.CONTINUOUS, name='b')
        c = m.addVars(self.labels, self.l_index, vtype=GRB.BINARY, name='c')
        d = m.addVars(self.b_index, vtype=GRB.BINARY, name='d')
        z_group = m.addVars(n_group, self.l_index, vtype=GRB.BINARY, name='z_group')
        l = m.addVars(self.l_index, vtype=GRB.BINARY, name='l')
        L = m.addVars(self.l_index, vtype=GRB.CONTINUOUS, name='L')

        M_pos = {}
        M_neg = {}
        for t in self.l_index:
            M_pos[t] = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"M_pos_{t}")
            M_neg[t] = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"M_neg_{t}")

        obj = gp.quicksum(L[t] for t in self.l_index) + self.alpha * gp.quicksum(d[t] for t in self.b_index)
        m.setObjective(obj)

        for t in self.l_index:
            m.addConstr(M_pos[t] == gp.quicksum(wpos_g[g] * z_group[g, t] for g in range(n_group)))
            m.addConstr(M_neg[t] == gp.quicksum(wneg_g[g] * z_group[g, t] for g in range(n_group)))

        BIGM = 1e6
        for t in self.l_index:
            m.addConstr(L[t] >= M_neg[t] - BIGM * (1 - c[1, t]))
            m.addConstr(L[t] <= M_neg[t] + BIGM * (1 - c[1, t]))
            m.addConstr(L[t] >= M_pos[t] - BIGM * (c[1, t]))
            m.addConstr(L[t] <= M_pos[t] + BIGM * (c[1, t]))

        m.addConstrs(z_group.sum(g, '*') == 1 for g in range(n_group))

        m.addConstrs(gp.quicksum(counts[g] * z_group[g, t] for g in range(n_group)) >= self.min_samples_split * l[t]
                     for t in self.l_index)

        m.addConstrs(c.sum('*', t) == l[t] for t in self.l_index)

        m.addConstrs(a.sum('*', t) == d[t] for t in self.b_index)
        m.addConstrs(b[t] <= d[t] for t in self.b_index)
        m.addConstrs(d[t] <= d[t // 2] for t in self.b_index if t != 1)

        min_dis = self._calMinDist(unique_x)
        max_min_dis = 1 + np.max(min_dis) if len(min_dis) > 0 else 1.0

        for t in self.l_index:
            left = (t % 2 == 0)
            ta = t // 2
            while ta != 0:
                if left:
                    m.addConstrs(
                        gp.quicksum(a[j, ta] * (unique_x[g, j] + min_dis[j]) for j in range(p))
                        + max_min_dis * (1 - d[ta])
                        <= b[ta] + max_min_dis * (1 - z_group[g, t])
                        for g in range(n_group)
                    )
                else:
                    m.addConstrs(
                        gp.quicksum(a[j, ta] * unique_x[g, j] for j in range(p))
                        >= b[ta] - (1 - z_group[g, t])
                        for g in range(n_group)
                    )
                left = (ta % 2 == 0)
                ta //= 2

        z = {}
        for i in range(n_orig):
            g = int(inv_idx[i])
            for t in self.l_index:
                z[(i, t)] = z_group[g, t]

        return m, a, b, c, d, l, z, M_pos, M_neg


    def _update_probabilities_from_solution(self, z_sol, w_pos, w_neg):

        n = z_sol.shape[0]
        leaf_probs = {}
        for idx, t in enumerate(self.l_index):
            Mp = 0.0
            Mn = 0.0
            for i in range(n):
                if z_sol[i, idx] == 1:
                    Mp += w_pos[i]
                    Mn += w_neg[i]
            denom = Mp + Mn
            leaf_probs[t] = (Mp / denom) if denom > 0 else 0.0
        return leaf_probs

    def predict_proba(self, X):
        if not self.trained:
            raise AssertionError('Model not trained yet.')

        if not self.leaf_pos_prob:
            leaf_prob_map = {t: (1.0 if self._c.get((1, t), 0) >= 0.5 else 0.0) for t in self.l_index}
        else:
            leaf_prob_map = self.leaf_pos_prob

        probs = []
        for xi in X / self.scales:
            t = 1
            while t not in self.l_index:
                right = (sum([self._a.get((j, t), 0) * xi[j] for j in range(self.p)]) + 1e-9 >= self._b.get(t, 0))
                if right:
                    t = 2 * t + 1
                else:
                    t = 2 * t
            probs.append(leaf_prob_map.get(t, 0.0))
        return np.array(probs)

    def print_tree_info(self):

        if not self.trained:
            print("Model not trained yet.")
            return

        print("=== Tree structure and parameters ===")
        print("Branch nodes:")
        for t in self.b_index:
            # decide which feature used
            feat = None
            for j in range(self.p):
                if self._a.get((j, t), 0) >= 0.5:
                    feat = j
                    break
            if feat is None:
                print(f" Node {t}: no split (d={self._d.get(t, 0):.3f})")
            else:
                print(
                    f" Node {t}: split on feature {feat}, threshold b={self._b.get(t, 0):.4f}, d={self._d.get(t, 0):.3f}")

        print("Leaf nodes:")
        for t in self.l_index:
            pred = None
            for k in self.labels:
                if self._c.get((k, t), 0) >= 0.5:
                    pred = k
                    break
            prob = self.leaf_pos_prob.get(t, None)
            print(f" Leaf {t}: predict={pred}, pos_prob={prob}")

    def _buildMIP(self, x, y):
        m = gp.Model('m')
        configure_solver_logging(
            m,
            show_raw_console=self.output,
            save_log=self.save_gurobi_log,
            log_file=self._solver_log_path(),
        )
        m.Params.timelimit = self.timelimit
        m.params.threads = 0

        m.modelSense = GRB.MINIMIZE

        a = m.addVars(self.p, self.b_index, vtype=GRB.BINARY, name='a') # splitting feature
        b = m.addVars(self.b_index, vtype=GRB.CONTINUOUS, name='b') # splitting threshold
        c = m.addVars(self.labels, self.l_index, vtype=GRB.BINARY, name='c') # node prediction
        d = m.addVars(self.b_index, vtype=GRB.BINARY, name='d') # splitting option
        z = m.addVars(self.n, self.l_index, vtype=GRB.BINARY, name='z') # leaf node assignment
        l = m.addVars(self.l_index, vtype=GRB.BINARY, name='l') # leaf node activation
        L = m.addVars(self.l_index, vtype=GRB.CONTINUOUS, name='L') # leaf node misclassified
        M = m.addVars(self.labels, self.l_index, vtype=GRB.CONTINUOUS, name='M') # leaf label counts
        N = m.addVars(self.l_index, vtype=GRB.CONTINUOUS, name='N') # leaf sample counts

        baseline = self._calBaseline(y)

        min_dis = self._calMinDist(x)

        obj = L.sum() / baseline + self.alpha * d.sum()
        m.setObjective(obj)

        m.addConstrs(L[t] >= N[t] - M[k,t] - self.n * (1 - c[k,t]) for t in self.l_index for k in self.labels)
        m.addConstrs(L[t] <= N[t] - M[k,t] + self.n * c[k,t] for t in self.l_index for k in self.labels)
        m.addConstrs(gp.quicksum((y[i] == k) * z[i,t] for i in range(self.n)) == M[k,t]
                                 for t in self.l_index for k in self.labels)
        m.addConstrs(z.sum('*', t) == N[t] for t in self.l_index)
        m.addConstrs(c.sum('*', t) == l[t] for t in self.l_index)
        for t in self.l_index:
            left = (t % 2 == 0)
            ta = t // 2
            while ta != 0:
                if left:
                    m.addConstrs(gp.quicksum(a[j,ta] * (x[i,j] + min_dis[j]) for j in range(self.p))
                                 +
                                 (1 + np.max(min_dis)) * (1 - d[ta])
                                 <=
                                 b[ta] + (1 + np.max(min_dis)) * (1 - z[i,t])
                                 for i in range(self.n))
                else:
                    m.addConstrs(gp.quicksum(a[j,ta] * x[i,j] for j in range(self.p))
                                 >=
                                 b[ta] - (1 - z[i,t])
                                 for i in range(self.n))
                left = (ta % 2 == 0)
                ta //= 2
        m.addConstrs(z.sum(i, '*') == 1 for i in range(self.n))
        m.addConstrs(z[i,t] <= l[t] for t in self.l_index for i in range(self.n))
        m.addConstrs(z.sum('*', t) >= self.min_samples_split * l[t] for t in self.l_index)
        m.addConstrs(a.sum('*', t) == d[t] for t in self.b_index)
        m.addConstrs(b[t] <= d[t] for t in self.b_index)
        m.addConstrs(b[t] >= 0 for t in self.b_index)
        m.addConstrs(d[t] <= d[t//2] for t in self.b_index if t != 1)

        return m, a, b, c, d, l

    def fit(self, x, y):

        self.n, self.p = x.shape
        if self.output:
            print('Training data include {} instances, {} features.'.format(self.n,self.p))

        # labels
        self.labels = np.unique(y)

        self.scales = np.max(x, axis=0)
        self.scales[self.scales == 0] = 1

        m, a, b, c, d, l = self._buildMIP(x/self.scales, y)
        if self.warmstart:
            self._setStart(x, y, a, c, d, l)

        self._current_iteration = None
        monitor = SolverMonitor(
            heartbeat_seconds=self.heartbeat_seconds,
            show_concise_progress=self.concise_progress,
            context=self.progress_context,
            sense=int(m.ModelSense),
        )
        m.optimize(combined_callback(monitor))
        if m.SolCount <= 0:
            self.last_solver_status = int(m.Status)
            raise RuntimeError(f"OCT solve returned no feasible solution (status={m.Status})")
        self.optgap = m.MIPGap
        record = solver_record(m, monitor=monitor)
        self.solve_history = [record]
        self.last_solver_status = record["status"]
        if self.iteration_callback is not None:
            self.iteration_callback(record, self)

        self._a = {ind:a[ind].x for ind in a}
        self._b = {ind:b[ind].x for ind in b}
        self._c = {ind:c[ind].x for ind in c}
        self._d = {ind:d[ind].x for ind in d}
        self._l = {ind:l[ind].x for ind in l}

        try:
            max_label = max(self.labels)
            pass
        except Exception:
            pass

        self.trained = True
        return self

    def predict(self, x):
        if not self.trained:
            raise AssertionError('Model not trained yet.')

        labelmap = {}
        for t in self.l_index:
            vals = [(k, self._c.get((k, t), 0)) for k in self.labels]
            k_best = max(vals, key=lambda kv: kv[1])[0]
            labelmap[t] = k_best

        y_pred = []
        for xi in x / self.scales:
            t = 1
            while t not in self.l_index:
                right = (sum([self._a.get((j, t), 0.0) * xi[j] for j in range(self.p)]) + 1e-9 >= self._b.get(t, 0.0))
                t = 2 * t + 1 if right else 2 * t
            y_pred.append(labelmap[t])
        return np.array(y_pred)


    @staticmethod
    def _calBaseline(y):
        mode = stats.mode(y, keepdims=True)[0][0]
        return np.sum(y == mode)

    @staticmethod
    def _calMinDist(x):
        min_dis = []
        for j in range(x.shape[1]):
            xj = x[:,j]
            xj = np.unique(xj)
            xj = np.sort(xj)[::-1]
            dis = [1]
            for i in range(len(xj)-1):
                dis.append(xj[i] - xj[i+1])
            min_dis.append(np.min(dis) if np.min(dis) else 1)
        return min_dis

    def _setStart(self, x, y, a, c, d, l):
        if self.min_samples_split > 1:
            clf = tree.DecisionTreeClassifier(max_depth=self.max_depth, min_samples_split=self.min_samples_split, random_state=self.random_state)
        else:
            clf = tree.DecisionTreeClassifier(max_depth=self.max_depth, random_state=self.random_state)
        clf.fit(x, y)

        rules = self._getRules(clf)

        for t in self.b_index:
            if rules[t].feat is None or rules[t].feat == tree._tree.TREE_UNDEFINED:
                d[t].start = 0
                for f in range(self.p):
                    a[f,t].start = 0
            else:
                d[t].start = 1
                for f in range(self.p):
                    if f == int(rules[t].feat):
                        a[f,t].start = 1
                    else:
                        a[f,t].start = 0

        for t in self.l_index:
            if rules[t].value is None:
                l[t].start = int(t % 2)
                if t % 2:
                    t_leaf = t
                    while rules[t].value is None:
                        t //= 2
                    for k in self.labels:
                        if k == np.argmax(rules[t].value):
                            c[k, t_leaf].start = 1
                        else:
                            c[k, t_leaf].start = 0
                else:
                    for k in self.labels:
                        c[k, t].start = 0
            else:
                l[t].start = 1
                for k in self.labels:
                    if k == np.argmax(rules[t].value):
                        c[k, t].start = 1
                    else:
                        c[k, t].start = 0

    def _getRules(self, clf):
        node_map = {1:0}
        for t in self.b_index:
            node_map[2*t] = -1
            node_map[2*t+1] = -1
            l = clf.tree_.children_left[node_map[t]]
            node_map[2*t] = l
            r = clf.tree_.children_right[node_map[t]]
            node_map[2*t+1] = r

        rule = namedtuple('Rules', ('feat', 'threshold', 'value'))
        rules = {}
        for t in self.b_index:
            i = node_map[t]
            if i == -1:
                r = rule(None, None, None)
            else:
                r = rule(clf.tree_.feature[i], clf.tree_.threshold[i], clf.tree_.value[i,0])
            rules[t] = r
        for t in self.l_index:
            i = node_map[t]
            if i == -1:
                r = rule(None, None, None)
            else:
                r = rule(None, None, clf.tree_.value[i,0])
            rules[t] = r

        return rules

    def get_tree_params(self):
        if not self.trained:
            raise AssertionError('This optimalDecisionTreeClassifier instance is not fitted yet.')
        config = {
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'alpha': self.alpha,
            'warmstart': self.warmstart,
            'timelimit': self.timelimit,
            'output': self.output
        }

        branch_split_rules = {}
        for t in self.b_index:
            split_feat = None
            for j in range(self.p):
                if self._a[(j, t)] >= 1e-2:
                    split_feat = j
                    break
            branch_split_rules[t] = (split_feat, round(self._b[t], 6))

        leaf_predict_label = {}
        for t in self.l_index:
            for k in self.labels:
                if self._c[(k, t)] >= 1e-2:
                    leaf_predict_label[t] = k
                    break

        # 树结构核心参数
        tree_structure = {
             'branch_node_ids': self.b_index,
            'leaf_node_ids': self.l_index,
            'branch_split_rules': branch_split_rules,
            'node_split_flag': {t: round(self._d[t]) for t in self.b_index},
            'leaf_predict_label': leaf_predict_label,
            'leaf_activation': {t: 1 if t in leaf_predict_label else 0 for t in self.l_index}
        }

        training_metrics = {
            'trained': self.trained,
            'mip_opt_gap': self.optgap,
            'n_samples': self.n if hasattr(self, 'n') else None,
            'n_features': self.p if hasattr(self, 'p') else None,
            'n_classes': len(self.labels) if hasattr(self, 'labels') else None
        }

        data_info = {
            'feature_scales': self.scales
        }

        return {
            'config': config,
            'tree_structure': tree_structure,
            'training_metrics': training_metrics,
            'data_info': data_info
        }
