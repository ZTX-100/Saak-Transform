from sklearn.decomposition import PCA
import numpy as np
import cPickle, gzip
from sklearn import svm
#import matplotlib.pyplot as plt
#from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.ensemble import RandomForestClassifier
#
f = gzip.open('./mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()
train_label = np.concatenate((train_set[1], valid_set[1]))
test_label = test_set[1]

# (dc,ac,-ac) -> (dc,ac)
def Unsign(train_data):
    filternum = (train_data.shape[3]-1)/2
    ta1 = np.concatenate((train_data[:,:,:,:1], train_data[:,:,:,1:filternum+1] - train_data[:,:,:,filternum+1:],),axis=3)
    return ta1.reshape(ta1.shape[0],-1)

# load features
if True:    
    train_data = np.load('./feature/train_before_f_test_' + '1_v' + '.npy')
    print (train_data.shape)
    ta = Unsign(train_data)
    print(np.count_nonzero(ta))
    train_data = np.load('./feature/train_before_f_test_' + '2_v' + '.npy')
    ta = np.concatenate((ta, Unsign(train_data)), 1)
    print (ta.shape)
    print(np.count_nonzero(ta))
    train_data = np.load('./feature/train_before_f_test_' + '3_v' + '.npy')
    ta = np.concatenate((ta, Unsign(train_data)), 1)
    print(np.count_nonzero(ta))
    print (ta.shape)
    train_data = np.load('./feature/train_before_f_test_' + '4_v' + '.npy')
    ta = np.concatenate((ta, Unsign(train_data)), 1)
    print(np.count_nonzero(ta))
    print (ta.shape)
    train_data = np.load('./feature/train_before_f_test_' + '5_v' + '.npy')
    ta = np.concatenate((ta, Unsign(train_data)), 1)
    print(np.count_nonzero(ta))
    print (ta.shape)
    num_traindata = 60000
    train_data = (ta[:num_traindata])
    #idx = train_data < 0
    #train_data[idx] = -train_data[idx]
    #train_label = train_label[:num_traindata]
    test_data = np.load('./feature/test_before_f_test_' + '1_v' + '.npy')
    ta = Unsign(test_data)
    print (ta.shape)
    test_data = np.load('./feature/test_before_f_test_' + '2_v' + '.npy')
    ta = np.concatenate((ta, Unsign(test_data)), 1)
    print (ta.shape)
    test_data = np.load('./feature/test_before_f_test_' + '3_v' + '.npy')
    ta = np.concatenate((ta, Unsign(test_data)), 1)
    print (ta.shape)
    test_data = np.load('./feature/test_before_f_test_' + '4_v' + '.npy')
    ta = np.concatenate((ta, Unsign(test_data)), 1)
    print (ta.shape)
    test_data = np.load('./feature/test_before_f_test_' + '5_v' + '.npy')
    ta = np.concatenate((ta, Unsign(test_data)), 1)
    print (ta.shape)
    test_data = (ta)
#idx = test_data < 0
#test_data[idx] = test_data[idx]
    del ta
import pickle
path = './CLF'
def load_obj(name, pathname = path):
    name = name + '.pkl'
    with open(os.path.join(pathname, name), 'rb') as f:
        return pickle.load(f)
        
def save_obj(obj, name ):    
    with open(path+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

# feature selection
def evac_ftest(rep2,label):              
    F,p = f_classif(rep2,label)
    low_conf = p>0.05
    F[low_conf] = 0
    where_are_NaNs = np.isnan(F)
    F[where_are_NaNs] = 0
    return F
def evac_RFS(feature,label):
    from skfeature.function.sparse_learning_based import RFS
    from skfeature.utility.sparse_learning import construct_label_matrix, feature_ranking

    Label = construct_label_matrix(label)
	# obtain the feature weight matrix
    Weight = RFS.rfs(feature, Label, gamma=0.1)

    # sort the feature scores in an ascending order according to the feature scores
    idx = feature_ranking(Weight)
    return idx
    # obtain the dataset on the selected features
#    selected_features = X[:, idx[0:num_fea]]
EVA = 'FTest'
save = False
if EVA == 'FTest':    
    print 'computer f-test:'    
    if True:
        Eva = evac_ftest(train_data, train_label)
        #np.save('./EVA/eva_ftest_v'+str(num)+'.npy',Eva)
    else: Eva = np.load('./EVA/eva_ftest_v'+str(num)+'.npy')
    Eva_cnt = Eva.shape[0]
    print Eva_cnt, np.count_nonzero(Eva)
    idx = Eva > np.sort(Eva)[::-1][int(np.count_nonzero(Eva)*0.75)-1]
    idx2 = np.argsort(Eva)[::-1]
    train_data = train_data[:,idx]
    test_data = test_data[:,idx]
if EVA == 'RFS':
    print 'computer RFS:'
    num_fea = 200
    if save == True:
        idx = evac_RFS(train_data, train_label)
        np.save('./EVA/eva_RFS_v'+str(num)+'.npy',idx)
    else:
        idx = np.load('./EVA/eva_RFS_v'+str(num)+'.npy')
    idx = idx[0:num_fea]
    
if EVA == 'MRMR':
    print 'computer MRMR:'
    num_fea = 200
    from skfeature.function.information_theoretical_based import MRMR
    if save == True:
        idx = MRMR.mrmr(train_data, train_label)#, n_selected_features=num_fea)
        np.save('./EVA/eva_MRMR_v'+str(num)+'.npy',idx)
    else:
        idx = np.load('./EVA/eva_MRMR_v'+str(num)+'.npy')  
    idx = idx[:num_fea]
if EVA == 'RF':
    print 'computer ReliefF:'
    num_fea = 200
    from skfeature.function.similarity_based import reliefF
    if save == True:
        score = reliefF.reliefF(train_data, train_label)
        # rank features in descending order according to score
        idx = reliefF.feature_ranking(score)
        np.save('./EVA/eva_RF_v'+str(num)+'.npy',idx)
    else:
        idx = np.load('./EVA/eva_RF_v'+str(num)+'.npy')
    idx = idx[:num_fea]
    
    train_data = train_data[:,idx]
    test_data = test_data[:,idx]

if False:
    import matplotlib.pyplot as plt
    
    for f_id in range(1):
        f_id = 14
        plt.figure(f_id)
        # the histogram of the data
        for class_id in range(10):
            x = train_data[train_label == class_id]
            y,binEdges=np.histogram(x[:,f_id],bins=20)
            bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
            plt.plot(bincenters,y,'-',label='class'+str(class_id))
            
#            plt.xlabel('Smarts')
#            plt.ylabel('Probability')
        plt.grid(True)
        plt.legend(loc=2)   
        plt.show()
    
if False:
    print 'computer svm:'       
    clf = svm.SVC()
    clf.fit(train_data, train_label)
    pre  = clf.predict(test_data)
    print np.count_nonzero(pre == test_label)
    
if True:
    # feature reduction
    print 'computer pca:' 
    pca = PCA(svd_solver='full')
    pca.fit(train_data)
    pca_k = pca.components_
    for n_components in [64]:
        pca.components_ = pca_k[:n_components,:]
        traindata = pca.transform(train_data)
        testdata = pca.transform(test_data)
        print 'computer svm:'  
        if False:
            clf = svm.SVC(probability=True)
            clf.fit(traindata, train_label)
            pre  = clf.predict_proba(testdata)
            preall = clf.predict_proba(traindata)
            
            
            print 'reduce dim to %d:' % n_components
            print np.count_nonzero(np.argsort(pre,1)[:,9] == test_label)
        else:
            clf = svm.SVC()
            clf.fit(traindata, train_label)
            pre  = clf.predict(testdata)
            print 'reduce dim to %d:' % n_components
            print np.count_nonzero(pre == test_label)
        # if True:
        #     for class_id in range(10):
        #         idx = test_label == class_id
        #         acc = float(np.count_nonzero(pre[idx] == test_label[idx]))/float(np.count_nonzero(idx))
        #         print 'class_id %d:' % class_id,' %f' % acc

if False:
    
    train_pre  = (preall)
    test_pre = (pre)
    
    train_pre_label = np.zeros(train_label.shape) + 1
    train_pre_label[np.argsort(train_pre,1)[:,9] == train_label] = 0
    train_pre_label[np.argsort(train_pre,1)[:,8] == train_label] = 1
    train_pre_label[np.argsort(train_pre,1)[:,7] == train_label] = 1
#    idx_train = train_pre_label != 10
#    train_pre_label = train_pre_label[idx_train]
#    train_pre = train_pre[idx_train]
#    pretest_p = np.sort(preall)[:,8:]
#    idx_error = pretest_p[idx_train][:,1]/(pretest_p[idx_train][:,0]+0.000001) < 10000
    
    test_pre_label = np.zeros(test_label.shape) + 1
    test_pre_label[np.argsort(test_pre,1)[:,9] == test_label] = 0
    test_pre_label[np.argsort(test_pre,1)[:,8] == test_label] = 1
    test_pre_label[np.argsort(test_pre,1)[:,7] == test_label] = 1
#    idx_test = test_pre_label != 10
#    test_pre_label = test_pre_label[idx_test]
#    test_pre = test_pre[idx_test]
    print 'svm 1:'
    weight={}
    weight[0] =  1
    weight[1] =  1000
    clf2 = svm.SVC(class_weight = 'balanced')
#    print 'computer pca:' 
#    pca = PCA(svd_solver='full')
#    pca.fit(np.sort(train_pre))
#    pca_k = pca.components_
#    for n_components in [3]:
#        pca.components_ = pca_k[:n_components,:]
#        trainpre = pca.transform((np.sort(train_pre)))
#        testpre = pca.transform((np.sort(test_pre)))
    clf2.fit(np.sort(train_pre)[:,5:], train_pre_label)    
    test_pre_pre  = clf2.predict(np.sort(test_pre)[:,5:])#test_pre_pre  = clf2.predict((np.sort(test_pre)))
    print np.count_nonzero(test_pre_pre == (test_pre_label))
    print np.count_nonzero(test_pre_pre == 0)
    np.count_nonzero(test_pre_label[test_pre_pre == 0] != 0)    