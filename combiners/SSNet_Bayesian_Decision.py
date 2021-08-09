'''
==================================================LICENSING TERMS==================================================
This code and data was developed by employees of the National Institute of Standards and Technology (NIST), an agency of the Federal Government. Pursuant to title 17 United States Code Section 105, works of NIST employees are not subject to copyright protection in the United States and are considered to be in the public domain. The code and data is provided by NIST as a public service and is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED OR STATUTORY, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST does not warrant or make any representations regarding the use of the data or the results thereof, including but not limited to the correctness, accuracy, reliability or usefulness of the data. NIST SHALL NOT BE LIABLE AND YOU HEREBY RELEASE NIST FROM LIABILITY FOR ANY INDIRECT, CONSEQUENTIAL, SPECIAL, OR INCIDENTAL DAMAGES (INCLUDING DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, AND THE LIKE), WHETHER ARISING IN TORT, CONTRACT, OR OTHERWISE, ARISING FROM OR RELATING TO THE DATA (OR THE USE OF OR INABILITY TO USE THIS DATA), EVEN IF NIST HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.
To the extent that NIST may hold copyright in countries other than the United States, you are hereby granted the non-exclusive irrevocable and unconditional right to print, publish, prepare derivative works and distribute the NIST data, in any medium, or authorize others to do so on your behalf, on a royalty-free basis throughout the world.
You may improve, modify, and create derivative works of the code or the data or any portion of the code or the data, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the code or the data and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the code or the data: Citation recommendations are provided below. Permission to use this code and data is contingent upon your acceptance of the terms of this agreement and upon your providing appropriate acknowledgments of NIST's creation of the code and data.
Paper Title:
    SSNet: a Sagittal Stratum-inspired Neural Network Framework for Sentiment Analysis
SSNet authors and developers:
    Apostol Vassilev:
        Affiliation: National Institute of Standards and Technology
        Email: apostol.vassilev@nist.gov
    Munawar Hasan:
        Affiliation: National Institute of Standards and Technology
        Email: munawar.hasan@nist.gov
    Jin Honglan
        Affiliation: National Institute of Standards and Technology
        Email: honglan.jin@nist.gov
====================================================================================================================
'''


res = ['max', 'avg', 'sum']
res_wt_maj = ['max', 'avg', 'sum', 'maj']

def _max(a, b):
    a_mx = max(a)
    a_idx = a.index(a_mx)

    b_mx = max(b)
    b_idx = b.index(b_mx)

    if a_mx >= b_mx:
        return a_idx
    else:
        return b_idx


def _avg(a, b):
    p0 = (a[0] + b[0]) / float(2)
    p1 = (a[1] + b[1]) / float(2)
    if p0 >= p1:
        return 0
    else:
        return 1


def _sum(a, b):
    p0 = a[0] + b[0]
    p1 = a[1] + b[1]
    if p0 >= p1:
        return 0
    else:
        return 1


def _resolve(a, b, c, d, resolve):
    idx_a = a.index(max(a))
    idx_b = b.index(max(b))
    idx_c = c.index(max(c))
    idx_d = d.index(max(d))

    ll = [a, b, c ,d]
    ll_max = [max(a), max(b), max(c), max(d)]

    if resolve == 'max':
        _max_ll_max = max(ll_max)
        _max_ll_max_idx = ll_max.index(_max_ll_max)

        _max_ll = ll[_max_ll_max_idx]
        _max_idx = _max_ll.index(_max_ll_max)

        return _max_idx

    if resolve == 'avg':
        p0 = (a[0] + b[0] + c[0] +d[0])/4
        p1 = (a[1] + b[1] + c[1] +d[1])/4

        avg_idx = [p0, p1].index(max([p0, p1]))
        return avg_idx
    
    if resolve == 'sum':
        p0 = (a[0] + b[0] + c[0] +d[0])
        p1 = (a[1] + b[1] + c[1] +d[1])

        avg_idx = [p0, p1].index(max([p0, p1]))
        return avg_idx

def mj2(x, y, p, resolve):
    correct_pred = 0
    wrong_pred = 0

    for idx in range(len(p)):
        ll = p[idx]
        file_name = ll[0]
        label = int(ll[1])

        x_proba = x[file_name]
        y_proba = y[file_name]

        if x_proba.index(max(x_proba)) == y_proba.index(max(y_proba)):
            if x_proba.index(max(x_proba)) == label:
                correct_pred += 1
            else:
                wrong_pred += 1
        else:
            for func_name, resolve_func in [('max', _max), ('avg', _avg), ('sum', _sum)]:
                if resolve == func_name:
                    resolved_idx = resolve_func(x_proba, y_proba)
                    if resolved_idx == label:
                        correct_pred = correct_pred + 1
                    else:
                        wrong_pred = wrong_pred + 1

    assert correct_pred + wrong_pred == len(p), "length mismatch"
    return round((float(correct_pred) / float(len(p))), 5) * float(100)


def mj3(x, y, z, p, resolve):
    correct_pred = 0
    wrong_pred = 0

    for idx in range(len(p)):
        ll = p[idx]
        file_name = ll[0]
        label = int(ll[1])

        x_proba = x[file_name]
        y_proba = y[file_name]
        z_proba = z[file_name]

        if resolve == 'max':
            ll = [x_proba, y_proba, z_proba]
            l_max = [max(x_proba), max(y_proba), max(z_proba)]
            _max_item = max(l_max)
            _max_item_index = l_max.index(_max_item)

            ll_max_index = ll[_max_item_index]
            max_idx = ll[_max_item_index].index(max(ll[_max_item_index]))
            
            if max_idx == label:
                correct_pred += 1
            else:
                wrong_pred += 1

        if resolve == 'avg':
            p0 = (x_proba[0] + y_proba[0] + z_proba[0]) / 3.
            p1 = (x_proba[1] + y_proba[1] + z_proba[1]) / 3.
            ll = [p0, p1]
            avg_idx = ll.index(max(ll))
            if avg_idx == label:
                correct_pred += 1
            else:
                wrong_pred += 1
        
        if resolve == 'sum':
            p0 = (x_proba[0] + y_proba[0] + z_proba[0])
            p1 = (x_proba[1] + y_proba[1] + z_proba[1])
            ll = [p0, p1]
            avg_idx = ll.index(max(ll))
            if avg_idx == label:
                correct_pred += 1
            else:
                wrong_pred += 1
        
        if resolve == 'maj':
            ix = x_proba.index(max(x_proba))
            iy = y_proba.index(max(y_proba))
            iz = z_proba.index(max(z_proba))
            
            if ix == iy == iz:
                if ix == label:
                    correct_pred += 1
                else:
                    wrong_pred += 1
            else:
                if ix == iy:
                    if ix == label:
                        correct_pred += 1
                    else:
                        wrong_pred += 1
                if ix == iz:
                    if ix == label:
                        correct_pred += 1
                    else:
                        wrong_pred += 1
                if iy == iz:
                    if iy == label:
                        correct_pred += 1
                    else:
                        wrong_pred += 1
        

    assert correct_pred + wrong_pred == len(p), "length mismatch"
    return round((float(correct_pred) / float(len(p))), 5) * float(100)


def mj4(x, y, z, v, p, resolve):
    correct_pred = 0
    wrong_pred = 0
    for idx in range(len(p)):
        ll = p[idx]
        file_name = ll[0]
        label = int(ll[1])

        x_proba = x[file_name]
        y_proba = y[file_name]
        z_proba = z[file_name]
        v_proba = v[file_name]

        index = _resolve(x_proba, y_proba, z_proba, v_proba, resolve)
        if index == label:
            correct_pred += 1
        else:
            wrong_pred += 1

    assert correct_pred + wrong_pred == len(p), "length mismatch"
    return round((float(correct_pred) / float(len(p))), 5) * float(100)


def bayesian_decision(tr_list, imdb_tr_list, te_list, imdb_te_list):
    
    for i in range(len(tr_list)):
        for k, v in tr_list[i].items():
            tr_list[i][k] = [float(1) - v, v]
    
    for i in range(len(te_list)):
        for k, v in te_list[i].items():
            te_list[i][k] = [float(1) - v, v]

    acc_dict = dict()
    for L, F, res_list in [(2, mj2, res), (3, mj3, res_wt_maj), (4, mj4, res)]:
        if len(tr_list) == L:
            for r in res_list:
                a1 = F(*tr_list[:L], imdb_tr_list, r)
                a2 = F(*te_list[:L], imdb_te_list, r)
                acc_dict[r] = [a1, a2]

    return acc_dict
