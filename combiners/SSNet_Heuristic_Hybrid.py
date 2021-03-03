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


def threshold_one_on_three(c, current_th, dict_list, clist, data, resolve):
    correct_pred = 0
    wrong_pred = 0
    current_dict = dict_list[c]
    for sample in data:
        file_name = sample[0]
        label = int(sample[1])
        #print(sample)
        proba = [current_dict[file_name][0], current_dict[file_name][1]]
        #print(max(proba), current_th)
        if max(proba) >= current_th:
            if proba.index(max(proba)) == label:
                correct_pred = correct_pred + 1
            else:
                wrong_pred = wrong_pred + 1
        else:
            x_proba = dict_list[clist[0]][file_name]
            y_proba = dict_list[clist[1]][file_name]
            z_proba = dict_list[clist[2]][file_name]

            if resolve == 'max':
                ll = [x_proba, y_proba, z_proba]
                l_max = [max(x_proba), max(y_proba), max(z_proba)]
                _max_item = max(l_max)
                _max_item_index = l_max.index(_max_item)

                ll_max_index = ll[_max_item_index]
                max_idx = ll[_max_item_index].index(max(ll[_max_item_index]))
                
                if max_idx == label:
                    correct_pred = correct_pred + 1
                else:
                    wrong_pred = wrong_pred + 1

            if resolve == 'avg':
                p0 = (x_proba[0] + y_proba[0] + z_proba[0]) / 3.
                p1 = (x_proba[1] + y_proba[1] + z_proba[1]) / 3.
                ll = [p0, p1]
                avg_idx = ll.index(max(ll))
                if avg_idx == label:
                    correct_pred = correct_pred + 1
                else:
                    wrong_pred = wrong_pred + 1
            
            if resolve == 'sum':
                p0 = (x_proba[0] + y_proba[0] + z_proba[0])
                p1 = (x_proba[1] + y_proba[1] + z_proba[1])
                ll = [p0, p1]
                avg_idx = ll.index(max(ll))
                if avg_idx == label:
                    correct_pred = correct_pred + 1
                else:
                    wrong_pred = wrong_pred + 1
            
            if resolve == 'maj':
                ix = x_proba.index(max(x_proba))
                iy = y_proba.index(max(y_proba))
                iz = z_proba.index(max(z_proba))
                
                if ix == iy == iz:
                    if ix == label:
                        correct_pred = correct_pred + 1
                    else:
                        wrong_pred = wrong_pred + 1
                else:
                    if ix == iy:
                        if ix == label:
                            correct_pred = correct_pred + 1
                        else:
                            wrong_pred = wrong_pred + 1
                    if ix == iz:
                        if ix == label:
                            correct_pred = correct_pred + 1
                        else:
                            wrong_pred = wrong_pred + 1
                    if iy == iz:
                        if iy == label:
                            correct_pred = correct_pred + 1
                        else:
                            wrong_pred = wrong_pred + 1

    assert correct_pred + wrong_pred == len(data), "length mismatch"
    return round((float(correct_pred) / float(len(data))), 5) * float(100)


# threshold with one base and one non-base
def threshold_one_on_one(c, current_th, dict_list, clist, data, comb_index):
    correct_pred = 0
    wrong_pred = 0
    current_dict = dict_list[c]
    for sample in data:
        file_name = sample[0]
        label = int(sample[1])
        #print(sample)
        proba = [current_dict[file_name][0], current_dict[file_name][1]]
        #print(max(proba), current_th)
        if max(proba) >= current_th:
            if proba.index(max(proba)) == label:
                correct_pred = correct_pred + 1
            else:
                wrong_pred = wrong_pred + 1
        else:
            #print("th1=>", c, comb_index)
            x_proba = dict_list[clist[comb_index]][file_name]

            if x_proba.index(max(x_proba)) == label:
                correct_pred = correct_pred + 1
            else:
                wrong_pred = wrong_pred + 1

    assert correct_pred + wrong_pred == len(data), "length mismatch"
    return round((float(correct_pred) / float(len(data))), 5) * float(100)


# threshold with one base and two non-base
def threshold_one_on_two(c, current_th, dict_list, clist, data, comb_index, resolve):
    correct_pred = 0
    wrong_pred = 0
    current_dict = dict_list[c]
    for sample in data:
        file_name = sample[0]
        label = int(sample[1])
        #print(sample)
        proba = [current_dict[file_name][0], current_dict[file_name][1]]
        #print(max(proba), current_th)
        if max(proba) >= current_th:
            if proba.index(max(proba)) == label:
                correct_pred = correct_pred + 1
            else:
                wrong_pred = wrong_pred + 1
        else:
            x_proba = dict_list[clist[0]][file_name]
            y_proba = dict_list[clist[1]][file_name]
            z_proba = dict_list[clist[2]][file_name]

            a = -1
            b = -1

            if comb_index == 0:
                a = x_proba
                b = y_proba
            if comb_index == 1:
                a = x_proba
                b = z_proba
            if comb_index == 2:
                a = y_proba
                b = z_proba

            if resolve == 'max':
                index = _max(a, b)
            if resolve == 'avg':
                index = _avg(a, b)
            if resolve == 'sum':
                index = _sum(a, b)

            if index == label:
                correct_pred = correct_pred + 1
            else:
                wrong_pred = wrong_pred + 1

    assert correct_pred + wrong_pred == len(data), "length mismatch"
    return round((float(correct_pred) / float(len(data))), 5) * float(100)


def heuristic_hybrid(tr_list, imdb_tr_list, te_list, imdb_te_list):
    
    result_dict = dict()

    for i in range(len(tr_list)):
        for k, v in tr_list[i].items():
            tr_list[i][k] = [float(1) - v, v]

    for i in range(len(te_list)):
        for k, v in te_list[i].items():
            te_list[i][k] = [float(1) - v, v]
    
    list1 = list()
    list2 = list()
    list3 = list()

    list_a = list()
    list_b = list()
    list_c = list()
    list_d = list()

    combinations = [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]
    models_name = ['model_{a}', 'model_{b}', 'model{c}', 'model_{d}']
    for i in range(len(combinations)):
        for j in range(3):
            ll_ac = []
            ll_th = []
            th = 0.5

            dd = dict()

            while True:
                _acc = threshold_one_on_one(
                    i, th, tr_list, combinations[i], imdb_tr_list, j)
                
                ll_ac.append(round(_acc, 5))
                ll_th.append(th)

                th = round(th + 0.01, 2)
                if th > 0.99:
                    break
            
            _max_acc_tr = max(ll_ac)
            _max_th_tr = ll_th[ll_ac.index(_max_acc_tr)]
            
            dd["tr_acc"] = _max_acc_tr

            _acc = threshold_one_on_one(i, _max_th_tr, te_list, combinations[i], imdb_te_list, j)
            a_th = [_max_acc_tr, _acc, _max_th_tr]

            dd["te_acc"] = _acc
            dd["th"] = _max_th_tr

            if i == 0:
                list_a.append(dd)
            if i == 1:
                list_b.append(dd)
            if i == 2:
                list_c.append(dd)
            if i == 3:
                list_d.append(dd)
    
    #print(list1)

    for i in range(len(combinations)):
        for j in range(3):
            for k in range(len(res)):
                dd = dict()
                ll_ac = []
                ll_th = []
                th = 0.5
                while True:
                    #c, current_th, dict_list, clist, data, comb_index, resolve
                    _acc = threshold_one_on_two(
                        i, th, tr_list, combinations[i], imdb_tr_list, j, res[k])
                    ll_ac.append(round(_acc, 5))
                    ll_th.append(th)

                    th = round(th + 0.01, 2)
                    if th > 0.99:
                        break
                _max_acc_tr = max(ll_ac)
                dd["tr_acc"] = _max_acc_tr
                _max_th_tr = ll_th[ll_ac.index(_max_acc_tr)]

                _acc = threshold_one_on_two(
                    i, _max_th_tr, te_list, combinations[i], imdb_te_list, j, res[k])
                dd["te_acc"] = _acc
                a_th = [_max_acc_tr, _acc, _max_th_tr]
                dd["th"] = _max_th_tr
                #list2.append(a_th)

                if i == 0:
                    list_a.append(dd)
                if i == 1:
                    list_b.append(dd)
                if i == 2:
                    list_c.append(dd)
                if i == 3:
                    list_d.append(dd)

    #print(list2)

    for i in range(len(combinations)):
        for r in res_wt_maj:
            ll_ac = []
            ll_th = []
            dd = dict()
            th = 0.5
            while True:
                _acc = threshold_one_on_three(
                    i, th, tr_list, combinations[i], imdb_tr_list, r)
                ll_ac.append(round(_acc, 5))
                ll_th.append(th)

                th = round(th + 0.01, 2)
                if th > 0.99:
                        break
            _max_acc_tr = max(ll_ac)
            dd["tr_acc"] = _max_acc_tr
            _max_th_tr = ll_th[ll_ac.index(_max_acc_tr)]

            _acc = threshold_one_on_three(
                i, _max_th_tr, te_list, combinations[i], imdb_te_list, r)
            dd["te_acc"] = _acc
            a_th = [_max_acc_tr, _acc, _max_th_tr]
            dd["th"] = _max_th_tr

            if i == 0:
                list_a.append(dd)
            if i == 1:
                list_b.append(dd)
            if i == 2:
                list_c.append(dd)
            if i == 3:
                list_d.append(dd)

        #list3.append(a_th)
    
    #print(list3)
    '''
    print("model_{a}: ")
    print(list_a)
    print("\nmodel_{b}: ")
    print(list_b)
    print("\nmodel_{c}: ")
    print(list_c)
    print("\nmodel_{d}: ")
    print(list_d)
    '''
    dd = dict()

    dd["model_{1}"] = list_a
    dd["model_{2}"] = list_b
    dd["model_{3}"] = list_c
    dd["model_{4}"] = list_d
    
    return dd
