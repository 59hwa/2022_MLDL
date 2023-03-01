
from datasets import BreastCancerDataset
from util import computeClassificationAcc
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np


def q7():
    from perceptron import Perceptron
    model = Perceptron()
    thresh = 0
    model.setThreshold(thresh)
    model.train(tr_x, tr_y)
    model.predict(val_x)
    return(pred_n_draw_graph(model, tr_x, tr_y, val_x, val_y, "q7"))

def sweep_q7():
    sweep_list = np.zeros((20))
    from perceptron import Perceptron
    model = Perceptron()
    pred = np.array(())
    for i in range(len(sweep_list)):
        sweep_list[i] = -i*500
        model.setThreshold(int(sweep_list[i]))
        model.train(tr_x, tr_y)
        j=pred_n_draw_graph(model, tr_x, tr_y, val_x, val_y, "q3")
        pred = np.append(pred,j)
    print(sweep_list)
    print(pred)
    plt.plot(np.log10(sweep_list),pred)
    plt.xlabel('log scale lam')
    plt.ylabel('acc')
    plt.show()
    
def sklearn_perceptron():
    from sklearn.linear_model import Perceptron 
    model = Perceptron(alpha=1.0,max_iter=1000, eta0=1)
    model.fit(tr_x, tr_y)
    y_hat = model.predict(val_x)
    acc = computeClassificationAcc(val_y, y_hat)
    print("sklearn_perceptron")
    print(acc)
    return acc

    
def pred_n_draw_graph(model, tra_x, tr_y, vala_x, val_y, title):
    ## Prediction
    y_hat = model.predict(vala_x)
    ## Compute Error
    acc = computeClassificationAcc(val_y, y_hat) 
    print(acc)
    return(acc)
   
    tr_x = tra_x[:,:2]
    val_x = vala_x[:,:2] 
    y_hat2 = y_hat
    ## Draw Graph
    fig = plt.figure()
    plt3d = fig.add_subplot(projection='3d')
    
    tr_true = (tr_y == 1)
    trt_x = tr_x[tr_true, 0]
    trt_y = tr_x[tr_true, 1]
    trt_z = tr_y[tr_true]
    trf_x = tr_x[~tr_true, 0]
    trf_y = tr_x[~tr_true, 1]
    trf_z = tr_y[~tr_true]
    
    val_true = (val_y == 1)
    valt_x = val_x[val_true, 0]
    valt_y = val_x[val_true, 1]
    valt_z = val_y[val_true]
    valf_x = val_x[~val_true, 0]
    valf_y = val_x[~val_true, 1]
    valf_z = val_y[~val_true]
    
    vpt = y_hat2[val_true][:, 0]
    vpf = y_hat2[~val_true][:, 0]
    
    dx = np.linspace(tr_x[:, 0].min(), tr_x[:, 0].max(), 100)
    dy = np.linspace(tr_x[:, 1].min(), tr_x[:, 1].max(), 100)
  

    plt3d.scatter(trt_x, trt_y, trt_z, color = 'r', alpha = 0.5, label="Train-True")
    plt3d.scatter(trf_x, trf_y, trf_z, color = 'b', alpha = 0.5, label="Train-False")
    plt3d.scatter(valt_x, valt_y, valt_z, color = 'g', alpha = 0.5, label="Valid-True")
    plt3d.scatter(valf_x, valf_y, valf_z, color = 'y', alpha = 0.5, label="Valid-False")
    plt3d.scatter(valt_x, valt_y, vpt, color = 'k', alpha = 0.5, label="Pred-True")
    plt3d.scatter(valf_x, valf_y, vpf, color = 'violet', alpha = 0.5, label="Pred-False")
    
    x_surf, y_surf = np.meshgrid(dx, dy)
   
    dz = []

    for xs, ys in zip(x_surf, y_surf):
        # print(np.shape(np.concatenate([xs.reshape(-1, 1), ys.reshape(-1, 1)], axis=1)))
        pred = model.predict2(np.concatenate([xs.reshape(-1, 1), 
                                              ys.reshape(-1, 1)], axis=1))
        dz.append(pred)
    dz = np.concatenate(dz, axis=1)
    plt3d.plot_surface(x_surf, y_surf, dz.T)
    plt.title(title)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    j=0.0
    k=0.0
    for i in range(20):
        dataset = BreastCancerDataset()
        tr_x, tr_y, val_x, val_y = dataset.getDataset_cls()
        print('Perceptron Acc epoch'+str(i)+'->')
        j=j+q7()
        k=k+sklearn_perceptron()
    print('ACC average of handmade perceprton model is '+ str(j/20))
    print('ACC average of sklearn perceptron model is '+ str(k/20))
    # from sklearn.linear_model import LogisticRegression
    # dataset = BreastCancerDataset()
    
    # tr_x, tr_y, val_x, val_y = dataset.getDataset_cls()
    # sweep_q7()
    # sklearn_perceptron()
    # test=q7()