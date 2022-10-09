with open("/nas/project_data/B1_Behavior/hakim/ME-GraphAU/data/DISFA/list/DISFA_train_label_fold3.txt", 'r') as fp:
    num_lines_b = sum(1 for line in fp)

with open("/nas/project_data/B1_Behavior/hakim/ME-GraphAU/data/BP4D/list/BP4D_train_label_fold3.txt", 'r') as fp:
    num_lines_d = sum(1 for line in fp)

print(num_lines_b + num_lines_d)

with open("/nas/project_data/B1_Behavior/hakim/ME-GraphAU/data/list/train_label_fold3.txt", 'r') as fp:
    num_lines_tr = sum(1 for line in fp)

print(num_lines_tr)
