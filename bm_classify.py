import numpy as np


def binary_train(X, y, loss, w0=None, b0=None, step_size=0.5, max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: binary training labels, a N dimensional numpy array where 
    N is the number of training points, indicating the labels of 
    training data
    - loss: loss type, either perceptron or logistic
    - step_size: step size (learning rate)
	- max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: D-dimensional vector, a numpy array which is the weight 
    vector of logistic or perceptron regression
    - b: scalar, which is the bias of logistic or perceptron regression
    """
    N, D = X.shape
    assert len(np.unique(y)) == 2


    w = np.zeros(D)
    if w0 is not None:
        w = w0
    
    b = 0
    if b0 is not None:
        b = b0
        
    
    
    '''  
    for k in range(N):
        if(y_new[k]==0):
            y_new[k]=-1
    '''
  
    if loss == "perceptron":
        ############################################
        # TODO 1 : Edit this if part               #
        #          Compute w and b here            #

        q=np.ones((N,1))
        x_new=np.concatenate((q,X),axis=1)
        w_new=np.insert(w,0,b)
        
        y_new=np.array([])
        for k in range(N):
            if(y[k]==0):
                y_new=np.append(y_new, -1)
            else:
                y_new = np.append(y_new, y[k])
                
        for i in range(max_iterations): 
            temp=np.zeros(D+1)
            for j in range(N):
                t=y_new[j]*np.dot(w_new,x_new[j].transpose())
                if(t<=0):
                    temp+=y_new[j]*x_new[j]
            w_new+=step_size*temp/N
        
        b=w_new[0]
        w=np.delete(w_new,0,axis=0)
        
        ############################################
        

    elif loss == "logistic":
        ############################################
        # TODO 2 : Edit this if part               #
        #          Compute w and b here            #
        q=np.ones((N,1))
        x_n=np.concatenate((q,X),axis=1)
        w_n=np.insert(w,0,b)

        for i in range(max_iterations):
            t1=sigmoid(np.dot(w_n,x_n.transpose()))-y
            u=step_size*np.dot(t1,x_n)/N
            w_n=w_n-u

        b=w_n[0]
        w=np.delete(w_n,0,axis=0)
            
        ############################################    

        

    else:
        raise "Loss Function is undefined."

    assert w.shape == (D,)
    return w, b

def sigmoid(z):
    
    """
    Inputs:
    - z: a numpy array or a float number
    
    Returns:
    - value: a numpy array or a float number after computing sigmoid function value = 1/(1+exp(-z)).
    """

    ############################################
    # TODO 3 : Edit this part to               #
    #          Compute value                   #
    
    value=1 / (1 + np.exp(-z))
    ############################################
    
    return value

def binary_predict(X, w, b, loss="perceptron"):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: D-dimensional vector, a numpy array which is the weight 
    vector of your learned model
    - b: scalar, which is the bias of your model
    - loss: loss type, either perceptron or logistic
    
    Returns:
    - preds: N dimensional vector of binary predictions: {0, 1}
    """
    N, D = X.shape
    
    if loss == "perceptron":
        ############################################
        # TODO 4 : Edit this if part               #
        #          Compute preds                   #
        preds = np.zeros(N)
        n=len(X)
        q=np.ones((n,1))
        x_new=np.concatenate((q,X),axis=1)
        w_new=np.insert(w,0,b)
        value=np.dot(w_new,x_new.transpose())
        for i,val in enumerate(value):
            if(val>0):
                preds[i]=1
            else:
                preds[i]=0

        ############################################
        

    elif loss == "logistic":
        ############################################
        # TODO 5 : Edit this if part               #
        #          Compute preds                   #
        preds = np.zeros(N)
        n=len(X)
        q=np.ones((n,1))
        x_new=np.concatenate((q,X),axis=1)
        w_new=np.insert(w,0,b)
        value=np.dot(w_new,x_new.transpose())
        sig=sigmoid(value)
        for i,val in enumerate(sig):
            if(val>=0.5):
                preds[i]=1
            else:
                preds[i]=0
        ############################################
        

    else:
        raise "Loss Function is undefined."
    

    assert preds.shape == (N,) 
    return preds



def multiclass_train(X, y, C,
                     w0=None, 
                     b0=None,
                     gd_type="sgd",
                     step_size=0.5, 
                     max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: multiclass training labels, a N dimensional numpy array where
    N is the number of training points, indicating the labels of 
    training data
    - C: number of classes in the data
    - gd_type: gradient descent type, either GD or SGD
    - step_size: step size (learning rate)
    - max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: C-by-D weight matrix of multinomial logistic regression, where 
    C is the number of classes and D is the dimensionality of features.
    - b: bias vector of length C, where C is the number of classes
    """

    N, D = X.shape

    w = np.zeros((C, D))
    if w0 is not None:
        w = w0
    
    b = np.zeros(C)
    if b0 is not None:
        b = b0

    np.random.seed(42)
    if gd_type == "sgd":
        ############################################
        # TODO 6 : Edit this if part               #
        #          Compute w and b                 #
        w = np.zeros((C, D))
        b = np.zeros((C,1))
        
        q=np.ones((N,1))
        x_n=np.concatenate((X,q),axis=1)
        
        w_n=np.concatenate((w,b),axis=1)
        
        def softmax(x):
            x = np.exp(x - np.amax(x))
            d = np.sum(x)
            return (np.transpose(x) / d).T
    
        y = np.eye(C)[y]
        for i in range(max_iterations):
            idx = np.random.randint(N)
            error=softmax(np.dot(w_n,x_n[idx]))-y[idx]
            error=error[:,None]
            u=x_n[idx]
            u=u[:,None]
            w_n-=step_size*(np.dot(error,u.T))
        
        b=w_n[:,D]
        w=np.delete(w_n,D,axis=1)    
        
        ############################################
        

    elif gd_type == "gd":
        ############################################
        # TODO 7 : Edit this if part               #
        #          Compute w and b                 #
        w = np.zeros((C, D))
        b = np.zeros(C)
        
        def softmax(x):
            x = np.exp(x - np.amax(x))
            d = np.sum(x, axis=1)
            return (np.transpose(x) / d).T
    
        y = np.eye(C)[y]
        for i in range(max_iterations):
            error = softmax((w.dot(X.T)).T + b) - y
            w_grad = error.T.dot(X) / N
            b_grad = np.sum(error, axis=0) / N
            w -= step_size * w_grad
            b -= step_size * b_grad
        ############################################
        

    else:
        raise "Type of Gradient Descent is undefined."
    

    assert w.shape == (C, D)
    assert b.shape == (C,)

    return w, b


def multiclass_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: weights of the trained multinomial classifier, C-by-D 
    - b: bias terms of the trained multinomial classifier, length of C
    
    Returns:
    - preds: N dimensional vector of multiclass predictions.
    Outputted predictions should be from {0, C - 1}, where
    C is the number of classes
    """
    N, D = X.shape
    ############################################
    # TODO 8 : Edit this part to               #
    #          Compute preds                   #
    preds = np.zeros(N)
    
    def softmax(x):
        x = np.exp(x - np.amax(x))
        d = np.sum(x, axis=1)
        return (np.transpose(x) / d).T
    
    preds = softmax((np.matmul(w,np.transpose(X))).T + b)
    preds = np.argmax(preds, axis=1)
    ############################################

    assert preds.shape == (N,)
    return preds




        