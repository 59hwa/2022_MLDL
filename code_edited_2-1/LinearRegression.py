from datasets import BreastCancerDataset
from linear import Linear
from util import computeAvgRegrMSError
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def sklearn_vanilla():
    model = LinearRegression()
    model.fit(tr_x, tr_y)
    y_hat = model.predict(val_x)
    err = computeAvgRegrMSError(val_y, y_hat)
    print("sklearn_vanilla ")
    print(err)
    return(err)
    

def sklearn_ridge():
    from sklearn.linear_model import Ridge
    dataset = BreastCancerDataset()
    tr_x, tr_y, val_x, val_y = dataset.getDataset_reg()
    model = Ridge(alpha=1e-10)
    model.fit(tr_x, tr_y)
    y_hat = model.predict(val_x)
    err = computeAvgRegrMSError(val_y, y_hat)
    print("sklearn_ridge ")
    print(err)
    return(err)
    
    
def q1():
    dataset = BreastCancerDataset()
    
    model = Linear()
    ## Vanilla Linear Regression Model (Closed form solution)
    model.train_CFS(tr_x, tr_y)
    return(pred_n_draw_graph(model, tr_x, tr_y, val_x, val_y, "q1"))
    
def q2():
    dataset = BreastCancerDataset()
    model = Linear()
    ## Ridge Regression Model (Closed form solution)
    lam = 1e-10
    model.setLam(lam)
    model.train_ridge_CFS(tr_x, tr_y)
    return(pred_n_draw_graph(model, tr_x, tr_y, val_x, val_y, "q2"))


def q3():
    dataset = BreastCancerDataset()
    model = Linear()
    ## Ridge Regression Model (Gradient descent algorithm)
    lam = 1e-10
    model.setLam(lam)
    eta = 1
    model.setEta(eta)
    model.train_ridge_GD(tr_x, tr_y)
    return(pred_n_draw_graph(model, tr_x, tr_y, val_x, val_y, "q3"))
    
    
def pred_n_draw_graph(model, tr_x, tr_y, val_x, val_y, title):
    ## Prediction
    y_hat = model.predict(val_x)
    y_hat2 = model.predict2(val_x[:,:2])
    ## Compute Error
    error = computeAvgRegrMSError(val_y, y_hat) 
    print(error)
    return error
    # ## Draw Graph
    # fig = plt.figure(figsize=(9, 6))
    # ax = fig.add_subplot(111, projection='3d')
    # tx = tr_x[:, 0]
    # ty = tr_x[:, 1]
    # tz = tr_y[:]
    
    # vx = val_x[:, 0]
    # vy = val_x[:, 1]
    # vz = val_y[:]
    # vp = y_hat2[:]
    
    # dx = np.linspace(tx.min(), tx.max(), 100)
    # dy = np.linspace(ty.min(), ty.max(), 100)
    # dz = model.predict2(np.concatenate([dx.reshape(-1, 1), 
    #                                     dy.reshape(-1, 1)], axis=1)).reshape(-1)
    
    # ax.plot(dx, dy, dz, "b-", label="Regressed Line")
    # ax.scatter(tx, ty, tz, color = 'r', alpha = 0.5, label="Train")
    # ax.scatter(vx, vy, vz, color = 'g', alpha = 0.5, label="Valid-Real")
    # ax.scatter(vx, vy, vp, color = 'y', alpha = 0.5, label="Valid-Pred")
    # plt.title(title)
    # plt.legend()
    # plt.show()
def sweep_q2():
    sweep_list = np.zeros((100))

    model = Linear()
    ## Ridge Regression Model (Gradient descent algorithm)
    pred = np.array(())

    for i in range(len(sweep_list)):
        sweep_list[i] = pow(10,-100+i)
        model.setLam(sweep_list[i])
        model.train_ridge_CFS(tr_x, tr_y)
        j=pred_n_draw_graph(model, tr_x, tr_y, val_x, val_y, "q3")
        pred = np.append(pred,j)
    print(sweep_list)
    print(pred)
    plt.plot(np.log10(sweep_list),np.log10(pred))
    plt.xlabel('log scale lambda')
    plt.ylabel('log scaleerror')
    plt.show()
def sweep_q3():
    sweep_list = np.zeros((9))

    model = Linear()
    ## Ridge Regression Model (Gradient descent algorithm)
    pred = np.array(())
    model.setLam(1e-10)
    for i in range(len(sweep_list)):
        sweep_list[i] = pow(2,i-4)
        model.setEta(sweep_list[i])
        model.train_ridge_GD(tr_x, tr_y)
        j=pred_n_draw_graph(model, tr_x, tr_y, val_x, val_y, "q3")
        pred = np.append(pred,j)
    print(sweep_list)
    print(pred)
    plt.plot(np.log(sweep_list),np.log10(pred))
    plt.xlabel('log scale lambda')
    plt.ylabel('log scaleerror')
    plt.show()
        
if __name__ == "__main__":
    k=0.0
    j=0.0
    for i in range(1000):
        
        from sklearn.linear_model import LinearRegression
        dataset = BreastCancerDataset()
        tr_x, tr_y, val_x, val_y = dataset.getDataset_reg()
        
        print('Ridge linear regression err epoch'+str(i)+'->')
    #     # j=j+q1()
        j=j+q2()
        k=k+sklearn_ridge()
    #     # j=j+q3()
    # from sklearn.linear_model import LinearRegression
    # dataset = BreastCancerDataset()
    # tr_x, tr_y, val_x, val_y = dataset.getDataset_reg()
    # # sweep_q2()
    # sweep_q3()
    
    print('Error mean of handmade model(CFS) is '+ str(j/1000))
    print('Error mean of sklearn model is '+ str(k/1000))
    # sklearn_vanilla()
    # sklearn_ridge()
    # q2()
    # q3()