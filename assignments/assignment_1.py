import logging
import numpy as np
from sympy import *
from sympy.solvers import solve

logging.basicConfig(level=logging.INFO, format='%(message)s')


def farkas_lemma(A, b, max_steps=999):
    eq_num, var_num = A.shape[0], A.shape[1]
    if eq_num != b.shape[0]:
        logging.error('\nError: A.shape[0] != b.shape[0]\n')
        exit()
    if eq_num != var_num:
        logging.warn('It may crash since eq_num != var_num ({0} != {1}). \n'.format(eq_num, var_num))

    # constraint the inequalities
    A, b = np.copy(A), np.copy(b)

    x = np.asarray([Symbol('x{0}'.format(_)) for _ in range(0, var_num)])
    x_idx = np.asarray([_ for _ in range(0, var_num)])

    extra_x = np.asarray([Symbol('x{0}'.format(_ + var_num)) for _ in range(0, var_num)])
    extra_x_idx = np.asarray([_ + var_num for _ in range(0, var_num)])

    Ax = np.dot(A, x)

    inequalities_str = ''
    for i in range(0, eq_num):
        inequalities_str += '    {0} < {1}\n'.format(Ax[i], b[i])
    logging.info('\nSolve inequalities: \n'
                 '__________________________________\n'
                 '{0}\n'.format(inequalities_str))

    # contract variables for solution
    cobasic = np.copy(x)
    cobasic_idx = np.copy(x_idx)

    basic = np.copy(extra_x)            # basic is left_eq
    basic_idx = np.copy(extra_x_idx)

    right_eq = b - Ax

    coefficient = - np.copy(A)

    cobasic_dict = {}
    for c in cobasic:
        cobasic_dict[c] = 0

    cur_step = 0
    while True:
        cur_step += 1

        step_str = ''
        for i in range(0, eq_num):
            step_str += '    {0} = {1}\n'.format(basic[i], right_eq[i])
        logging.info('\n|Step-{0}: |\n'
                     '__________________________________\n'
                     '(a) Constrct: \n{1}'.format(cur_step, step_str))


        # set cobasic_dict to zeros to get basic_value
        basic_value = np.zeros(basic.shape)
        for i in range(0, eq_num):
            print(right_eq[i], cobasic_dict.keys())
            basic_value[i] = right_eq[i].subs(cobasic_dict)
        logging.info('(b) Substitute: \n    {0} = {1} => {2} = {3}\n'.format(cobasic, np.zeros(cobasic.shape), basic, basic_value))

        # find the first negative value in basic_value
        first_neg_eq_idx = next((i for i, x in enumerate(basic_value) if x < 0), None)
        
        # all values in basic_value are positive => feasible
        if first_neg_eq_idx is None:
            logging.info('\nInequalities are feasible. \n')
            
            final_eq = []
            basic_dict = {}
            for i in range(0, eq_num):
                basic_dict[basic[i]] = basic_value[i]
            for i in range(0, eq_num):
                final_eq.append((right_eq[i] - basic[i]).subs(basic_dict))

            solved = solve(final_eq)
            solution_str = ''
            for var in x:
                if var in solved.keys():
                    solution_str += '    {0} = {1}\n'.format(var, solved[var])
                elif var in basic_dict.keys():
                    solution_str += '    {0} = {1}\n'.format(var, basic_dict[var])
            logging.info('One solution is: \n{0}\n'.format(solution_str))            
            break        

        # still solving
        else:
            # find the first positive value of coefficient in current right eq
            first_pos_coef_idx = next((i for i, x in enumerate(coefficient[first_neg_eq_idx]) if x > 0), None) 
            
            # all coefficients in current right eq are negative => unfeasible
            if first_pos_coef_idx is None:
                logging.info('\nInequalities are unfeasible. \n')

                final_coefficient = []
                for ex_idx, ex_x in zip(extra_x_idx, extra_x):
                    if ex_idx in cobasic_idx:
                        final_coefficient.append(float(-coefficient[first_neg_eq_idx][np.where(cobasic_idx == ex_idx)]))
                    elif ex_idx == basic_idx[first_neg_eq_idx]:
                        final_coefficient.append(1)
                    else:
                        final_coefficient.append(0)
                final_coefficient = np.asarray(final_coefficient) 

                left = np.sum(np.dot(A, x) * final_coefficient)
                right = np.sum(b * final_coefficient)
                logging.info('No solution because: \n    {0} < {1}\n'.format(left, right))            
                break
            
            # still solving
            else:
                logging.info('(c) Extract: \n    variable - {0} - in eq. {1}\n'
                             .format(cobasic[first_pos_coef_idx], first_neg_eq_idx + 1))

                # update right_eq
                tmp_eq = (basic[first_neg_eq_idx] - right_eq[first_neg_eq_idx])
                tmp_eq = tmp_eq / coefficient[first_neg_eq_idx][first_pos_coef_idx]
                tmp_eq = tmp_eq.subs(cobasic[first_pos_coef_idx], 0)

                for i in range(0, eq_num):
                    if i == first_neg_eq_idx:
                        right_eq[i] = tmp_eq
                    else:
                        right_eq[i] = right_eq[i].subs(cobasic[first_pos_coef_idx], tmp_eq)

                # update basic/cobasic and sort them
                idx_cobasic_to_basic = cobasic_idx[first_pos_coef_idx]
                idx_basic_to_cobasic = basic_idx[first_neg_eq_idx]

                basic_idx[first_neg_eq_idx] = idx_cobasic_to_basic
                cobasic_idx[first_pos_coef_idx] = idx_basic_to_cobasic

                value_cobasic_to_basic = cobasic[first_pos_coef_idx]
                value_basic_to_cobasic = basic[first_neg_eq_idx]

                basic[first_neg_eq_idx] = value_cobasic_to_basic
                cobasic[first_pos_coef_idx] = value_basic_to_cobasic

                cobasic_idx, cobasic = zip(*sorted(zip(cobasic_idx, cobasic)))
                basic_idx, basic, right_eq = zip(*sorted(zip(basic_idx, basic, right_eq)))

                cobasic_idx, cobasic = np.asarray(cobasic_idx), np.asarray(cobasic)
                basic_idx, basic = np.asarray(basic_idx), np.asarray(basic)
                right_eq = np.asarray(right_eq)

                # update cobasic_dict
                cobasic_dict = {}
                for c in cobasic:
                    cobasic_dict[c] = 0

                # update coefficient
                for i in range(0, eq_num):
                    tmp_eq = right_eq[i] - right_eq[i].subs(cobasic_dict)
                    print(tmp_eq)
                    for j in range(0, var_num):
                        cobasic_dict[cobasic[j]] = 1
                        coefficient[i][j] = tmp_eq.subs(cobasic_dict)
                        cobasic_dict[cobasic[j]] = 0
                    print(coefficient)


if __name__ == '__main__':
    #-----------
    # example-1
    #-----------
    # A = np.array(
    #     [
    #         [-1, -2, 1],
    #         [1, -3, -1],
    #         [-1, -2, 2],
    #     ])
    # b = np.array([-1, 2, -2])

    #-----------
    # example-2
    #-----------
    # A = np.array(
    #     [
    #         [-1, 2, 1],
    #         [3, -2, 1],
    #         [-1, -6, -23],
    #     ])
    # b = np.array([3, -17, 19])


    A = np.random.uniform(-10, 10, (5, 5))
    b = np.random.uniform(-10, 10, (5))


    farkas_lemma(A, b)
