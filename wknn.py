# -*- coding: utf-8 -*- 

"""

P. Bugata, P. Drotar, Weighted nearest neighbors feature selection,
Knowledge-Based Systems (2018), doi:https://doi.org/10.1016/j.knosys.2018.10.004

"""

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
#from sklearn.preprocessing import OneHotEncoder    # needed for multi-class classification
from sklearn.base import TransformerMixin
import time

class WkNNFeatureSelector(TransformerMixin):
    
    def __init__(self, max_features, n_iters = 1000, n_iters_in_loop = 100, 
                 metric = 'euclidean', p = None, kernel = 'rbf', 
                 error_type = 'mse', delta = 1.0, 
                 lambda0 = 0.001, lambda1 = 0.001, lambda2 = 0.001, alpha = 100,
                 optimizer = 'SGD', learning_rate = 0.1, 
                 normalize_gradient = True, data_type = 'float32', scaling=True,
                 apply_weights = False, n_iters_weights = 300                                 
        ):
        # this class implements TransformerMixin interface
        TransformerMixin.__init__(self)
        
        # how many features to select
        self.max_features_ = max_features 
        # number of epochs
        self.n_iters_ = n_iters
        # number of iterations to display
        self.n_iters_in_loop_ = n_iters_in_loop
        # distance metric
        self.metric_ = metric
        # p norm used in minkowski distance
        self.p_ = p
        # kernel - distance evaluation function
        self.kernel_ = kernel
        # error function
        self.error_type_ = error_type
        # delta - used only for Huber Loss function
        self.delta_ = delta
        # regularization parameter for pseudo L0 regularization
        self.lambda0_ = lambda0
        # regularization parameter for L1 regularization
        self.lambda1_ = lambda1
        # regularization parameter for L2 regularization
        self.lambda2_ = lambda2
        # alpha - used only for pseudo L0 regularization
        self.alpha_ = alpha
        # optimizer type
        self.optimizer_ = optimizer
        # learning rate
        self.learning_rate_ = learning_rate
        # gradient normalization flag
        self.normalize_gradient_ = normalize_gradient
        # data type for precision and numerical stability
        self.data_type_ = data_type
        # standardization of input data
        self.scl_ = None
        if scaling:
            self.scl_ = StandardScaler()
        
        # selected features
        self.selected_features_ = None
        # selected features after fine-tuning weights
        self.final_selected_features_ = None
        
        # feature weights
        self.weights_ = None
        # feature weights after fine-tuning weights
        self.final_weights_ = None       
        
        # error
        self.error_ = None
        # error after fine-tuning weights
        self.final_error_ = None
    
        # checking parameters
        # unsupported 
        if metric not in ['euclidean', 'cityblock', 'minkowski']:
            raise ValueError('Unsupported metric')
        if kernel not in ['rbf', 'exp']:
            raise ValueError('Unsupported kernel')
        if error_type not in ['mse', 'mae', 'huber', 'ce']:
            raise ValueError('Unsupported error type')
        if optimizer not in ['SGD', 'Adam', 'Nadam', 'Adagrad', 'Adadelta', 'RMSProp', 'Momentum']:
            raise ValueError('Unsupported optimizer')
        if data_type not in ['float32', 'float64']:
            raise ValueError('Unsupported data type')
        # unallowed    
        if error_type == 'huber' and delta is None:            
            raise ValueError('Parameter delta for Huber function is missing.')
        if  alpha is None:            
            raise ValueError('Parameter alpha for L0 regularization is missing.')
        if optimizer != 'SGD' and self.normalize_gradient_:
            raise ValueError('Gradient normalization is alloved only for SGD.')

        # constant for numerical stability
        if data_type == 'float32':
            self.epsilon_ = tf.constant(1e-14, dtype='float32')
        elif data_type == 'float64':    
            self.epsilon_ = tf.constant(1e-300, dtype='float64')
            
        # classification task flag for multi-class classification
        self.classification = None
        
        # flag to decide whether apply feature weights when transforming data
        self.apply_weights = apply_weights
        
        # number of epochs to fine-tuning weights
        self.n_iters_weights_ = n_iters_weights

    # pairwise sqeuclidean distance
    def sqeuclidean_dist(self, A, B):  
        # using matrix multiplication for efficient computation of Euclidean distance
        with tf.variable_scope('squeclidean_dist'):
          if not B is None:        
              norm_a = tf.reduce_sum(tf.multiply(tf.square(A), self.weights_), 1)
              norm_b = tf.reduce_sum(tf.multiply(tf.square(B), self.weights_), 1)
              norm_a = tf.reshape(norm_a, [-1, 1])
              norm_b = tf.reshape(norm_b, [1, -1])
              scalar_product = tf.matmul(tf.multiply(A, self.weights_), B, False, True) 
          else:
              norm_a = tf.reduce_sum(tf.multiply(tf.square(A), self.weights_), 1)
              norm_b = norm_a
              norm_a = tf.reshape(norm_a, [-1, 1])
              norm_b = tf.reshape(norm_b, [1, -1])
              scalar_product = tf.matmul(tf.multiply(A, self.weights_), A, False, True) 
        
          D = norm_a - 2*scalar_product + norm_b

          # special modification of distance matrix - use small value instead of 0
          # to avoid computational problems when computing derivative
          D = tf.where(D > self.epsilon_, D, tf.ones_like(D)*self.epsilon_)                 
        
        return D

    # minkowski distance using map 
    def minkowski_dist_map(self, A, B):
    
        with tf.variable_scope('minkowski_dist'):
            if B is None:
                B = A
             
            def dist_(x):
                result = tf.abs(tf.subtract(x, B))
                    
                if self.p_ != 1:
                     result = tf.pow(result, self.p_)
                    
                result = tf.reduce_sum(tf.multiply(result, self.weights_), 1)        
                    
                if self.p_ != 1:                    
                    # special modification of distance matrix - use small value instead of 0
                    # to avoid computational problems when computing derivative
                    result = tf.where(result > self.epsilon_, result, tf.ones_like(result)*self.epsilon_)                 
                    
                    result = tf.pow(result, 1/self.p_)
                    
                return tf.transpose(result)
            
            
            D = tf.map_fn(fn=dist_, elems=A)
                    
        return D


    # minkowski distance using expand_dim (slower then map_fn)
    def minkowski_dist_expand(self, A, B):
    
        with tf.variable_scope('manhattan_dist'):            
            if B is None:
                B = A
    
            D = tf.abs(tf.subtract(A, tf.expand_dims(B, 1)))
    
            if self.p_ != 1:
                D = tf.pow(D, self.p_)
    
            D = tf.reduce_sum(tf.multiply(D, self.weights_), axis=2)        
    
            if self.p_ != 1:
                # special modification of distance matrix - use small value instead of 0
                # to avoid computational problems when computing derivative
                D = tf.where(D > self.epsilon_, D, tf.ones_like(D)*self.epsilon_)                 
                
                D = tf.pow(D, 1/self.p_)
                    
        return D

    # using of nested map fn - slow!
    def manhattan_dist(self, A, B):
    
        with tf.variable_scope('minkowski_dist'):
            if B is None:
                B = A
             
            def dist_(x):
                
                def dist_row_(y):
                    result = tf.abs(tf.subtract(x, y))
                    result = tf.reduce_sum(tf.multiply(result, self.weights_), 1)        
                    return result

                return tf.transpose(tf.map_fn(fn=dist_row_, elems=B))    
            
            D = tf.map_fn(fn=dist_, elems=A)
                    
        return D

    # computing similarity matrix using corresponding metric and kernel
    def similarity_matrix(self, A, B):
    
        similarity_mat = None
        if (self.metric_ == 'euclidean'   and self.kernel_ == 'rbf'):            
            dists = self.sqeuclidean_dist(A, B)
            similarity_mat = tf.exp(-dists)
    
        if (self.metric_ == 'euclidean'   and self.kernel_ == 'exp'):           
            dists = self.sqeuclidean_dist(A, B)                               
            dists_sqrt = tf.sqrt(dists)
            similarity_mat = tf.exp(-dists_sqrt)        
    
        if self.metric_ == 'cityblock':        
            self.p_ = 1
            dists = self.minkowski_dist_map(A, B)
            if self.kernel_ == 'exp':
                similarity_mat = tf.exp(-dists)
            elif self.kernel_ == 'rbf':     
                similarity_mat = tf.exp(-tf.square(dists))
    
        if self.metric_ == 'minkowski':        
            if self.p_ is None:
                raise ValueError('Parameter p is missing')
            dists = self.minkowski_dist_map(A, B)                        
            if self.kernel_ == 'exp':
                similarity_mat = tf.exp(-dists)
            elif self.kernel_ == 'rbf':     
                similarity_mat = tf.exp(-tf.square(dists))
    
        if similarity_mat is None:
            raise ValueError('Unsupported metric/kernel combination.')      
    
        return similarity_mat

    # compute prediction vector
    # prediction for i-th dataset point is weighted average of target values of other points
    def predict(self, similarity_matrix, y):
        zero_mat = tf.zeros_like(similarity_matrix)
        diag_mat = tf.diag(tf.ones(y.shape.dims[0].value, dtype=y.dtype))
        diag_mask = tf.greater(diag_mat, zero_mat)
        mod_sim_matrix = tf.where(diag_mask, zero_mat, similarity_matrix)
        
        weight_sums = tf.reshape(tf.reduce_sum(mod_sim_matrix, 1), [-1,1])            

        assert_op = tf.Assert(tf.reduce_all(weight_sums >= self.epsilon_), 
                              [tf.reduce_min(weight_sums)], name='assert_min_weigths') 

        with tf.control_dependencies([assert_op]):
          predictions = tf.divide(tf.matmul(mod_sim_matrix, y), weight_sums)
                   
        return predictions  

    # compute mean error - MSE, MAE, Huber loss or cross entropy
    def compute_error(self, y, y_pred):
        if self.error_type_ == 'mse':
            losses = tf.square(tf.subtract(y, y_pred)) 
        elif self.error_type_ == 'mae':
            losses = tf.abs(tf.subtract(y, y_pred))        
        elif self.error_type_ == 'huber':
            errors = tf.abs(tf.subtract(y, y_pred))
            hlf_1 = 0.5 * tf.square(tf.subtract(y, y_pred))
            hlf_2 = self.delta_ * errors - 0.5*self.delta_*self.delta_
            losses = tf.where(tf.less_equal(errors, self.delta_), hlf_1, hlf_2)                 
        elif self.error_type_ == 'ce':
            # log(1 - abs(y - y_pred))
            err_clipped = tf.maximum(1 - tf.abs(tf.subtract(y, y_pred)), self.epsilon_)
            losses = -tf.log(err_clipped)        

        # if y is one hot encoded, error is sum over all classes
        if self.classification:
            losses = tf.reduce_sum(losses, axis = 1)
                            
        return tf.reduce_mean(losses)

    # pseudo L0 regularization
    # sum of sigmoid functions indicating non-zero variable weight
    def pseudo_l0_regularization(self):
        reg = self.weights_
        reg = tf.exp(reg*(-self.alpha_))
        reg = tf.add(reg, tf.ones_like(reg))
        reg = tf.divide(tf.ones_like(reg), reg)
        reg = tf.add(reg, tf.ones_like(reg)*-1/2)
        reg = tf.reduce_sum(reg)*2
        return reg
    
    # L1 regularization to penalize sum of weights
    def l1_regularization(self):
        return tf.reduce_sum(self.weights_)
    
    # L2 regularization to penalize sum of squares of weights
    def l2_regularization(self):
        return tf.reduce_sum(tf.square(self.weights_))
 

    # normalize gradient to max decrease by 1*learning_rate
    def normalize_grad(self, grad, val):
        # gradient L1 norm
        gradient_norm = tf.reduce_sum(tf.abs(grad)) 

        # correction for negative weights
        new_val = tf.subtract(val, grad * self.learning_rate_)
        neg_sum = tf.reduce_sum(tf.minimum(new_val, 0.0))
        gradient_norm = tf.add(gradient_norm, neg_sum / self.learning_rate_)
        
        # max 100x increase
        gradient_norm = tf.maximum(gradient_norm, 0.01)

        normalized_grad = tf.divide(grad, gradient_norm)                        
        return normalized_grad

        
    # gradient clipping
    def modified_minimize(self, optimizer, cost_function):
        gvs = optimizer.compute_gradients(cost_function, var_list = [self.weights_])

        if self.normalize_gradient_:
            capped_gvs = [(self.normalize_grad(grad, val), val) for grad,val in gvs if grad is not None]                
        else:
            capped_gvs = [(tf.clip_by_norm(grad, 1.0), val) for grad,val in gvs if grad is not None]    
            
        apply_op = optimizer.apply_gradients(capped_gvs, name="apply_gradients")    
        
        # after applying gradient clip negative weights
        with tf.control_dependencies([apply_op]):      
            assign_op = self.weights_.assign(tf.maximum(self.weights_, 0))
            
        return assign_op

    # computing cost function
    def compute_cost(self, X, y):
        
        # applying distance evaluation function
        similarity_matrix_op = self.similarity_matrix(X, None)
        
        # predict target values
        y_pred = self.predict(similarity_matrix_op, y)
        
        # compute error
        mean_error = self.compute_error(y, y_pred)
    
        return mean_error


    # optimizer loop in computation graph        
    def optimizer_loop(self, optimizer, X, y, n_iter):
    
        # Use a resource variable for a true "read op"
        with tf.variable_scope("foo", reuse=tf.AUTO_REUSE):
            var_err = tf.get_variable(name="var_last_err", shape=[], use_resource=True, dtype=y.dtype)
            var_reg0 = tf.get_variable(name="var_last_L0reg", shape=[], use_resource=True, dtype=y.dtype)
            var_reg1 = tf.get_variable(name="var_last_L1reg", shape=[], use_resource=True, dtype=y.dtype)
            var_reg2 = tf.get_variable(name="var_last_L2reg", shape=[], use_resource=True, dtype=y.dtype)
        
        def _cond(i, _):
            return tf.less(i, n_iter)  
    
        def _body(i, sequencer):
    
            mean_error = self.compute_cost(X, y)
            modified_err = var_err.assign(mean_error)
            
            # add regularization   
            if self.lambda0_ != 0 and self.lambda1_ != 0 and self.lambda2_ != 0:
                reg0 = self.pseudo_l0_regularization()
                reg1 = self.l1_regularization()
                reg2 = self.l2_regularization()
                modified_reg0 = var_reg0.assign(reg0)
                modified_reg1 = var_reg1.assign(reg1)
                modified_reg2 = var_reg2.assign(reg2)
                cost = tf.add(mean_error, reg0 * self.lambda0_)
                cost = tf.add(cost, reg1 * self.lambda1_)                                        
                cost = tf.add(cost, reg2 * self.lambda2_)                                        
            elif self.lambda0_ != 0 and self.lambda1_ != 0:
                reg0 = self.pseudo_l0_regularization()
                reg1 = self.l1_regularization() 
                modified_reg0 = var_reg0.assign(reg0)
                modified_reg1 = var_reg1.assign(reg1)
                modified_reg2 = var_reg2.assign(0)
                cost = tf.add(mean_error, reg0 * self.lambda0_)
                cost = tf.add(cost, reg1 * self.lambda1_)                
            elif self.lambda0_ != 0 and self.lambda2_ != 0:
                reg0 = self.pseudo_l0_regularization()
                reg2 = self.l2_regularization() 
                modified_reg0 = var_reg0.assign(reg0)
                modified_reg1 = var_reg1.assign(0)
                modified_reg2 = var_reg2.assign(reg2)
                cost = tf.add(mean_error, reg0 * self.lambda0_)
                cost = tf.add(cost, reg2 * self.lambda2_)                
            elif self.lambda1_ != 0 and self.lambda2_ != 0:
                reg1 = self.l1_regularization()
                reg2 = self.l2_regularization()
                modified_reg0 = var_reg0.assign(0)
                modified_reg1 = var_reg1.assign(reg1)
                modified_reg2 = var_reg2.assign(reg2)
                cost = tf.add(mean_error, reg1 * self.lambda1_)
                cost = tf.add(cost, reg2 * self.lambda2_)                
            elif self.lambda0_ != 0:
                reg0 = self.pseudo_l0_regularization() 
                modified_reg0 = var_reg0.assign(reg0)
                modified_reg1 = var_reg1.assign(0)
                modified_reg2 = var_reg2.assign(0)
                cost = tf.add(mean_error, reg0 * self.lambda0_)                
            elif self.lambda1_ != 0:
                reg1 = self.l1_regularization() 
                modified_reg0 = var_reg0.assign(0)
                modified_reg1 = var_reg1.assign(reg1)
                modified_reg2 = var_reg2.assign(0)
                cost = tf.add(mean_error, reg1 * self.lambda1_)                
            elif self.lambda2_ != 0:
                reg2 = self.l2_regularization() 
                modified_reg0 = var_reg0.assign(0)
                modified_reg1 = var_reg1.assign(0)
                modified_reg2 = var_reg2.assign(reg2)
                cost = tf.add(mean_error, reg2 * self.lambda2_)
            else:                
                cost = mean_error
                modified_reg0 = var_reg0.assign(0)
                modified_reg1 = var_reg1.assign(0)
                modified_reg2 = var_reg2.assign(0)
                    
            train_op = self.modified_minimize(optimizer, cost)
                        
            with tf.control_dependencies([train_op, modified_err, modified_reg0, modified_reg1, modified_reg2]):
                next_sequencer = tf.ones([])
            return i + 1, next_sequencer
        
        init_err = var_err.assign(0.0)
        init_reg0 = var_err.assign(0.0)
        init_reg1 = var_err.assign(0.0)
        init_reg2 = var_err.assign(0.0)
        with tf.control_dependencies([init_err, init_reg0, init_reg1, init_reg2]):
            _, sequencer = tf.while_loop(cond=_cond, body=_body, loop_vars=[0, 1.], parallel_iterations=1)
    
        with tf.control_dependencies([sequencer]):
            last_err = var_err.read_value()
            last_reg0 = var_reg0.read_value()
            last_reg1 = var_reg1.read_value()
            last_reg2 = var_reg2.read_value()
    
        return last_err, last_reg0, last_reg1, last_reg2

    # creating optimizer
    def create_optimizer(self):       
        if self.optimizer_ == 'SGD':
            return tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate_)
        elif self.optimizer_ == 'Adam':
            return tf.train.AdamOptimizer(learning_rate=self.learning_rate_)
        elif self.optimizer_ == 'Nadam':
            return tf.contrib.opt.NadamOptimizer(learning_rate=self.learning_rate_)
        elif self.optimizer_ == 'Adagrad':
            return tf.train.AdagradOptimizer(learning_rate=self.learning_rate_)
        elif self.optimizer_ == 'Adadelta':
            return tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate_)
        elif self.optimizer_ == 'RMSProp':
            return tf.train.RMSPropOptimizer(learning_rate=self.learning_rate_)
        elif self.optimizer_ == 'Momentum':
            return tf.train.MomentumOptimizer(learning_rate=self.learning_rate_, momentum=0.9, use_nesterov=True)
                
    # building model - select variables and determining their weights    
    def fit(self, X, y, init_weights=None):

        tf.logging.set_verbosity(tf.logging.DEBUG)    
        
        # data standardization
        if self.scl_ is not None:
            X_scl = self.scl_.fit_transform(X)
        else:
            X_scl = X

        # create tensorflow graph
        data_type = self.data_type_
        
        # classification flag is automatically set according to number of y columns      
        self.classification = len(y[0]) > 1                   
        
        X_var = tf.constant(X_scl.astype(data_type), dtype=data_type, name='input_matrix')        
        y_var = tf.constant(y.astype(data_type), dtype=data_type, name='target_var')

        m = len(X[0])        
        if init_weights is None:        
            self.weights_ = tf.Variable(tf.ones([1,m], dtype=data_type)*1/m, 
                                        dtype=data_type, name='feature_weights')
        else:
            self.weights_ = tf.Variable(tf.convert_to_tensor(init_weights.astype(data_type)), 
                                        dtype=data_type, name='feature_weights')
        
        # creating and running optimizer
        optimizer = self.create_optimizer()
        optimizer_run = self.optimizer_loop(optimizer, X_var, y_var, self.n_iters_in_loop_)
        
        cost_op = self.compute_cost(X_var, y_var)        
        reg_op0 = self.pseudo_l0_regularization()    
        reg_op1 = self.l1_regularization()    
        reg_op2 = self.l2_regularization()    
                            
        end_opt = False
        
        # creating session and evaluation
        config = tf.ConfigProto()
        # don't pre-allocate memory
        config.gpu_options.allow_growth = True
        # create a session with specified option
        with tf.Session(config=config) as sess:    
            print('Start')
    
            # global variable initialization
            sess.run(tf.global_variables_initializer())
            
            error = None
            if self.n_iters_  < self.n_iters_in_loop_:
                self.n_iters_in_loop_ = self.n_iters_
            steps = (int) (self.n_iters_ / self.n_iters_in_loop_)
            
            for e in range(steps):
                try:                
                    error, reg0, reg1, reg2 = sess.run(optimizer_run)                  
                    print('Epoch:', (e+1) * self.n_iters_in_loop_, 'error:', error, 'L0 reg:', reg0, 'L1 reg:', reg1, 'L2 reg:', reg2)
    
                except Exception as ex:
                    print('Exception:', ex.__class__, ex.__context__)
                    end_opt = True

                if (end_opt or error is None or np.isnan(error)):
                    break
                                                           
            # print final error
            if not end_opt:
                if self.lambda0_ != 0 and self.lambda1_ != 0 and self.lambda2_ != 0:   
                    error, reg0, reg1, reg2 = sess.run([cost_op, reg_op0, reg_op1, reg_op2])
                    print('Final:', (e+1) * self.n_iters_in_loop_, 'error:', error, 'L0 reg:', reg0, 'L1 reg:', reg1, 'L2 reg:', reg2)
                elif self.lambda0_ != 0 and self.lambda1_ != 0:   
                    error, reg0, reg1 = sess.run([cost_op, reg_op0, reg_op1])
                    print('Final:', (e+1) * self.n_iters_in_loop_, 'error:', error, 'L0 reg:', reg0, 'L1 reg:', reg1)
                elif self.lambda0_ != 0 and self.lambda2_ != 0:   
                    error, reg0, reg2 = sess.run([cost_op, reg_op0, reg_op2])
                    print('Final:', (e+1) * self.n_iters_in_loop_, 'error:', error, 'L0 reg:', reg0, 'L2 reg:', reg2)
                elif self.lambda1_ != 0 and self.lambda2_ != 0:   
                    error, reg1, reg2 = sess.run([cost_op, reg_op1, reg_op2])
                    print('Final:', (e+1) * self.n_iters_in_loop_, 'error:', error, 'L1 reg:', reg1, 'L2 reg:', reg2)
                elif self.lambda0_ != 0:   
                    error, reg0 = sess.run([cost_op, reg_op0])
                    print('Final:', (e+1) * self.n_iters_in_loop_, 'error:', error, 'L0 reg:', reg0)
                elif self.lambda1_ != 0:   
                    error, reg1 = sess.run([cost_op, reg_op1])
                    print('Final:', (e+1) * self.n_iters_in_loop_, 'error:', error, 'L1 reg:', reg1)
                elif self.lambda2_ != 0:   
                    error, reg2 = sess.run([cost_op, reg_op2])
                    print('Final:', (e+1) * self.n_iters_in_loop_, 'error:', error, 'L2 reg:', reg2)
                else:
                    error = sess.run(cost_op)
                    print('Final:', (e+1) * self.n_iters_in_loop_, 'error:', error)
                self.error_ = error

            # final variable weights
            self.weights_ = np.abs(self.weights_.eval()).flatten()
            
            # release resources
            sess.close()

            self.selected_features_ = self.weights_.argsort()[::-1]                         
            nonzero_count = len(self.weights_[self.weights_ > 0])
            
            print('Non-zero weights: ', nonzero_count)
            print('Big weights: ', len(self.weights_[self.weights_ > 0.001]))
            print('Weight sum: ', self.weights_.sum())
            
            self.selected_features_ = self.selected_features_[:min(self.max_features_, nonzero_count)] 

            # additional fine-tuning weights
            if not self.apply_weights:
                return

            weights = self.weights_
            selected_features = self.selected_features_
            n_iters = self.n_iters_
            scl = self.scl_
            error = self.error_
            
            self.n_iters_ = self.n_iters_weights_
            self.scl_ = None
            self.apply_weights = False
            X_transformed = X_scl[:,selected_features]
            print('Selected features: ', selected_features.tolist())            
            print('Selected weights: ', weights[selected_features].tolist())            
            print('Selected weights sum: ', weights[selected_features].sum())            
            self.fit(X_transformed, y, weights[selected_features])
            self.final_selected_features_ = selected_features[self.selected_features_]
            self.final_weights_ = np.zeros(shape=m, dtype=weights.dtype)
            self.final_weights_[self.final_selected_features_] = self.weights_[self.selected_features_]
            self.final_error_ = self.error_
            print('Final selected features: ', self.final_selected_features_.tolist())            
            print('Final selected weights: ', self.final_weights_[self.final_selected_features_].tolist())            
                        
            self.weights_ = weights
            self.selected_features_ = selected_features
            self.scl_ = scl
            self.n_iters_ = n_iters
            self.apply_weights = True
            self.error_ = error
                                   

    # transforming data by variable selection and/or applying weights
    def transform(self, X):
        # data standardization
        if self.scl_ is not None:
            X_transformed = self.scl_.transform(X)
        else:
            X_transformed = X

        # applying variable weights to transformed data
        if self.apply_weights:
            X_transformed = X_transformed[:,self.final_selected_features_] 
            # correct application of weights according to metric                        
            weights_to_use = self.final_weights_[self.final_selected_features_]
            if self.metric_.count('minkowski') > 0 and self.p_ > 1:
                weights_to_use = weights_to_use**(1/self.p_)
            elif self.metric_.count('euclidean') > 0:
                weights_to_use = np.sqrt(weights_to_use)
            X_transformed = np.multiply(X_transformed, weights_to_use)
        else:
            X_transformed = X_transformed[:,self.selected_features_]             
        
        return X_transformed
            
                
if __name__ == '__main__':    
    # because of reproducible results
    tf.set_random_seed(1)
    
    # path to data
    path = 'D:/Users/pBugata/data/madelon'
    dataset_name = 'madelonHD'
    delim = None

    # loading data
    X = np.loadtxt(path + '/' + dataset_name + '_X.txt', delimiter=delim)
    y = np.loadtxt(path + '/' + dataset_name + '_y.txt', delimiter=delim).reshape(-1,1)
            
    # options for initial weights
    m = len(X[0])
    v_m = np.ones([1,m])*1/m
    v_zero = np.zeros([1,m])

    # necessary to uncomment for multi-class classification
    #y = OneHotEncoder(sparse=False).fit_transform(y)    

    # set parameters of transformer

    # max_features - number of features to select
    # n_iters - number of iterations (epochs)
    # n_iters_in_loop - number of iterations to display progress
    # metric - definition of distance (euclidean, cityblock, minkowski)
    # p - parameter for Minkowski distance
    # kernel - distance evaluation function (rbf, exp)
    # error_type - error function (mse, mae, huber, ce)
    # delta - parameter of Huber loss function
    # lambda0, lambda1, lambda2 - regularization parameter for pseudo L0, L1, and L2 regularization
    # alpha - parameter for pseudo L0 regularization (steepness of sigmoid function)
    # optimizer - optimizer type (SGD, Adam, Nadam, Adagrad, Adadelta, RMSProp, Momentum)
    # learning_rate - parameter for gradient descent
    # normalize_gradient - flag for L1 normalization of gradient
    # data_type - data type for controlling precision
    # scaling - flag for standardization of input data
    # apply_weights - flag for using weights for transformation of entire dataset
    # n_iters_weights - number of iterations for fine-tuning weights
    
    transformer = WkNNFeatureSelector(
                    max_features = 30, n_iters =10000, n_iters_in_loop = 10, 
                    metric = 'euclidean', p = 2, kernel = 'exp', 
                    error_type = 'ce', delta = 0.1, 
                    lambda0 = 0.00, lambda1 = 0.00, lambda2 = 0.00, alpha = 100,
                    optimizer = 'SGD', learning_rate = 0.1, 
                    normalize_gradient = True, data_type = 'float64', scaling = True,
                    apply_weights=False, n_iters_weights=1)                                 

    t_start = time.time()    

    # X, y - input data
    # init_weights - set initial weights
    transformer.fit(X, y, init_weights=v_m)

    col_indices = transformer.selected_features_    
    feature_weights = transformer.weights_[col_indices]
    print('Variables: ', col_indices.tolist())
    print('Variable weights: ', feature_weights.tolist())

    t_end = time.time()    
    print("Duration:", t_end - t_start)

    # creating and saving transformed data for future use (not necessary)
    X_transformed = transformer.transform(X)  
    np.savetxt(X=X_transformed, fname=path + '/' + dataset_name + '_transformed_X.txt', delimiter=' ')
    