n_sub_iter=pd.read_csv("variance_svrg_vA_swimmer_sn4.csv")

n_sub_iter2=pd.read_csv("variance_svrg_vA_swimmer_sn3.csv")

n_sub_iter = [n_sub_iter2, n_sub_iter]

n_s = pd.concat(n_sub_iter,axis=1)

n_s.to_csv("variance_svrg_vA_swimmer_sn.csv",index=False)