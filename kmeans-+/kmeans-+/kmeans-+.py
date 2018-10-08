import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

def Gain(centers, X, alpha = 0.75):
    SSEDM_Arr = []
    n_clusters = len(centers)
    kmeans2 = KMeans(n_clusters)
    kmeans2.cluster_centers_ = centers
    labels = kmeans2.predict(X)
    for i in range(n_clusters):
        SSEDM_Arr.append(np.sum(kmeans2.transform(X[labels == i]), axis = 0)[i])
    SSEDM_Arr = [SSEDM_Arr[i] * alpha for i in range(len(SSEDM_Arr))]
    return SSEDM_Arr

def select_max_Gain(centers, X, alpha = 3/4):
    SSEDM_Arr = Gain(centers, X, alpha = alpha)
    indices = np.argsort(-np.array(SSEDM_Arr))
    for i in range(len(centers)):
        maxIndex = indices[i]
        if maxIndex not in indivisible:
            return maxIndex, SSEDM_Arr[maxIndex]
        elif maxIndex == len(centers) / 2:
            maxIndex = None
            return maxIndex, None

def Cost(centers, X):
    cost_Arr = []
    n_clusters = len(centers)
    kmeans = KMeans(n_clusters)
    kmeans.cluster_centers_ = centers
    labels = kmeans.predict(X)
    for i in range(n_clusters):
        dis = np.sort(kmeans.transform(X[labels == i]),axis= 1)
        sub_SSEDM = np.sum(dis[:,0])
        ccp_SSEDM = np.sum(dis[:,1])
        cost_Arr.append(sub_SSEDM - ccp_SSEDM)
    return cost_Arr

def select_min_Cost(centers, X, max_Gain_index, maxGain):
    cost_Arr = Cost(centers, X)
    indices = np.argsort(np.array(cost_Arr))
    for i in range(len(centers)):
        minIndex = indices[i]
        if minIndex not in irremoveable and minIndex != max_Gain_index:
            return minIndex, cost_Arr[minIndex] 
        elif minIndex == len(centers) / 2:
            minIndex = None
            return minIndex, None
        return None, None

#index2 是否是index1的近邻
def adjacent(centers,index1, index2, X):
    n_clusters = len(centers)
    kmeans1 = KMeans(n_clusters)
    kmeans1.cluster_centers_ = centers
    labels = kmeans1.predict(X)
    if index2 in np.argsort(kmeans.transform(X[labels == index1]), axis = 1)[:,1]:
        return True
    return False

def strong_adjacent(centers, index1, X):
    result = []
    for i in range(len(centers)):
        if i == index1:
            pass
        elif adjacent(centers,index1, i, X):
            result.append(i)
    return result

def init_Kmeans(X):
    #init S 初始化一个聚类结果S，包含3个簇
    kmeans = KMeans(n_clusters = 3, init='random', max_iter=1)
    kmeans.fit(X)
    # print(kmeans.labels_)           #聚类的结果
    print(kmeans.cluster_centers_)  #聚类的中心
    # print(kmeans.transform(X))
    # print("初始的SSEDM为",kmeans.inertia_)      #所有数据点到距其最近中心的距离和SSEDM
    print("初始的SSEDM为",238.0)      #所有数据点到距其最近中心的距离和SSEDM
    cluster_centers = np.array([[6.8        ,3.04545455 ,5.62727273 ,2.01818182],
                        [5.83928571 ,2.73571429 ,4.33928571 ,1.40714286],
                        [5.7        ,2.8        ,4.5        ,1.3       ]])    #3*3
    return kmeans, cluster_centers

if __name__ == "__main__":
    success = 0
    k = 3
    unmatchable = []     #不匹配的簇对儿 索引
    irremoveable = []    #不可移除的簇  索引
    indivisible = []     #不可分割的簇  索引

    #load dataSet  150个样例，3类
    iris = load_iris()
    X = iris.data
    y = iris.target
    # print(X.shape,y.shape)
    # print(y)
    kmeans, cluster_centers = init_Kmeans(X)
    current_cluster_centers = np.copy(cluster_centers)
    # SSEDM = kmeans.inertia_
    SSEDM = 238.0
    # print(cluster_centers)
    while success <= k/2:  
    # while success <= 20:  
        print("gogogogo")
        max_Gain_index, maxGain = select_max_Gain(cluster_centers, X)          #找到Gain值最大的簇中心索引     要分割的
        if max_Gain_index is None:
            print("max gain is none")
            break
        min_Cost_index, minCost = select_min_Cost(cluster_centers, X, max_Gain_index, maxGain)          #找到Cost值最小的簇中心索引     要移除的
        if min_Cost_index is None:
            print("min cost is none")
            break
        unmatchable = [] 
        irremoveable = [] 
        indivisible = []
        strong_adjacent_before_min_Cost_indices = strong_adjacent(cluster_centers, min_Cost_index, X) #获取在簇中心移除之前的强近邻
        indivisible.extend(strong_adjacent_before_min_Cost_indices)        #把簇中心移除之前的强近邻加入到不可分割的簇
        #改变簇中心
        current_cluster_centers[min_Cost_index] = X[kmeans.labels_ == max_Gain_index][success]
        #重新聚类
        kmeans.cluster_centers_ = current_cluster_centers
        kmeans.labels_ = kmeans.predict(X)
        new_SSEDM = 0.0  #new_SSEDM
        for i in range(len(current_cluster_centers)):
            # print(kmeans.transform(X[kmeans.labels_ == i]))
            # print(np.sum(kmeans.transform(X[kmeans.labels_ == i]), axis = 0))
            # print("**************"*5)
            new_SSEDM += np.sum(kmeans.transform(X[kmeans.labels_ == i]), axis = 0)[i]
        # print("============="*5)
        print(SSEDM, new_SSEDM)
        if new_SSEDM > SSEDM:
            unmatchable.append([max_Gain_index, min_Cost_index])    #将当前的簇对儿标记为不匹配
        else:
            irremoveable.append(max_Gain_index)   #标记当前的簇对儿为不可以移除的簇
            irremoveable.append(min_Cost_index)
            strong_adjacent_max_Gain_indices = strong_adjacent(cluster_centers, min_Cost_index, X) + strong_adjacent(cluster_centers, max_Gain_index, X)
            irremoveable.extend(strong_adjacent_max_Gain_indices) #添加簇中心更改后的簇对儿的强近邻到不可移除的集合
            cluster_centers = current_cluster_centers
            SSEDM = new_SSEDM
            success += 1
            print("+1")
      
    print("最后结果为")
    print(cluster_centers) 
    print(SSEDM)
   