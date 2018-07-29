from __future__ import division
# -*- coding: utf-8 -*-

# Peter Bugata, Peter Drotár
# Weighted nearest neighbors feature selection

import numpy as np
import math as math
from scipy.spatial.distance import cdist
from sklearn.cross_validation import StratifiedKFold
import time as time

class NeighborsVarSelector(object):
    # parametre konstruktora:
    # lambda_const - konstanta lambda pred regularizacnym clenom
    # eta - nezaporne realne cislo, ktore nasobi gradient
    # batch_size - pocet nahodne vyberanych vzoriek pre SGD
    # n_iters - pocet iteracii algoritmu
    # metric - definicia vzdialenosti
    # error_type - miera chyby
    # eps - pre test na zastavenie
    # prec - presnost pri rieseni zaokruhlovacich chyb
    def __init__(self, lambda_const, eta, batch_size, n_iters, metric, p=None,
                 dist_weights='exp', error_type='mse', delta=0.1, eps=0.0001, prec=1e-10, 
                 c=1,check_neg_var_weights='zero', normalize_grad=False, stratified_SGD=False):
        self.lambda_const = lambda_const
        self.eta = eta
        self.batch_size = batch_size
        self.n_iters = n_iters
        self.eps = eps
        self.prec = prec
        
        # priznak pre optimalizaciu cez vektorovy vypocet
        self.vector_computation = True
        
        # regularizacia L1 alebo L0
        self.reg_method = 'L0'
        
        # priznak, ci sa ma SGD vykonavat so stratifikovanym vyberom
        self.stratified_SGD = stratified_SGD
        
        # minimalna pripustna suma vazenych vzdialenosti susedov
        self.min_weight_sum = 1e-16

        # konstanta pre vzdialenost
        self.c = c 
        
        # mocnina - pouzite iba pri vseobecnej 'minkowski' metrike
        self.p = p
        
        # delta - pouzite iba pri Huberovaj funkcii na vypocet chyby
        self.delta = delta
        
        # vektor vah vysvetlujucich premennych
        self.var_weights = None
        # distancna matica
        self.distance_matrix = None
        # vazena distancna matica
        self.wdistance_matrix = None
        # vektor predikcii
        self.predictions = None
        # vektor sum vah susedov pre jednotlive pozorovania
        self.weight_sums = None
        # boolovsky priznak debug - umoznuje potlacit, resp zobrazit debugovacie vypisy 
        self.debug = False
        # boolovsky priznak pre podrobnejsi vypis
        self.detail = False
        # retazcova premenna, ktora urcuje sposob korekcie vah
        self.check_neg_var_weights = check_neg_var_weights
        # boolovsky priznak, ktory urcuje, ci sa ma normalizovat gradient
        self.normalize_grad = normalize_grad
        # definicia vzdialenosti a hodnotiaca funkcia vzdialenosti
        if dist_weights == 'exp': 
            self.dist_weight_function = self.dist_weights_exp
            if metric == 'euclidean':
                self.dpi_dvl = self.dpi_dvl_euclid_exp
            elif metric == 'euclidean_mod':
                self.dpi_dvl = self.dpi_dvl_euclidmod_exp                
            elif metric == 'sqeuclidean':
                self.dpi_dvl = self.dpi_dvl_sqe_exp
            elif metric == 'sqeuclidean_mod':
                self.dpi_dvl = self.dpi_dvl_sqemod_exp            
            elif metric == 'cityblock':
                self.dpi_dvl = self.dpi_dvl_city_exp
            elif metric == 'minkowski':
                self.dpi_dvl = self.dpi_dvl_min_exp
            elif metric == 'minkowski_mod':
                self.dpi_dvl = self.dpi_dvl_minmod_exp                
            else:
                raise ValueError('Invalid distance definition.')
        elif dist_weights == 'expp': 
            self.dist_weight_function = self.dist_weights_expp
            if metric == 'euclidean':
                self.dpi_dvl_euclid_expp
            elif metric == 'euclidean_mod':
                self.dpi_dvl_euclidmod_expp                
            elif metric == 'sqeuclidean':
                self.dpi_dvl = self.dpi_dvl_sqe_expp
            elif metric == 'sqeuclidean_mod':
                self.dpi_dvl = self.dpi_dvl_sqemod_expp            
            elif metric == 'cityblock':
                self.dpi_dvl = self.dpi_dvl_city_expp
            elif metric == 'minkowski':
                self.dpi_dvl = self.dpi_dvl_min_expp 
            elif metric == 'minkowski_mod':
                self.dpi_dvl = self.dpi_dvl_minmod_expp                                
            else:
                raise ValueError('Invalid distance definition.')
        elif dist_weights == 'ax': 
            self.dist_weight_function = self.dist_weights_ax
            if metric == 'euclidean':
                self.dpi_dvl = self.dpi_dvl_euclid_ax            
            elif metric == 'euclidean_mod':
                self.dpi_dvl = self.dpi_dvl_euclidmod_ax
            elif metric == 'sqeuclidean':
                self.dpi_dvl = self.dpi_dvl_sqe_ax
            elif metric == 'sqeuclidean_mod':
                self.dpi_dvl = self.dpi_dvl_sqemod_ax            
            elif metric == 'cityblock':
                self.dpi_dvl = self.dpi_dvl_city_ax
            elif metric == 'minkowski':
                self.dpi_dvl = self.dpi_dvl_min_ax                                
            elif metric == 'minkowski_mod':
                self.dpi_dvl = self.dpi_dvl_minmod_ax                
            else:
                raise ValueError('Invalid distance definition.')                
        elif dist_weights == 'inv':
            self.dist_weight_function = self.dist_weights_inv
            if metric == 'euclidean':
                self.dpi_dvl = self.dpi_dvl_euclid_inv            
            elif metric == 'euclidean_mod':
                self.dpi_dvl = self.dpi_dvl_euclidmod_inv
            elif metric == 'sqeuclidean':
                self.dpi_dvl = self.dpi_dvl_sqe_inv
            elif metric == 'sqeuclidean_mod':
                self.dpi_dvl = self.dpi_dvl_sqemod_inv            
            elif metric == 'cityblock':
                self.dpi_dvl = self.dpi_dvl_city_inv
            elif metric == 'minkowski':
                self.dpi_dvl = self.dpi_dvl_min_inv
            elif metric == 'minkowski_mod':
                self.dpi_dvl = self.dpi_dvl_minmod_inv                                
            else:
                raise ValueError('Invalid distance definition.')
        elif dist_weights == 'loginv':
            self.dist_weight_function = self.dist_weights_loginv
            if metric == 'euclidean':
                self.dpi_dvl = self.dpi_dvl_euclid_loginv
            elif metric == 'euclidean_mod':
                self.dpi_dvl = self.dpi_dvl_euclidmod_loginv                
            elif metric == 'sqeuclidean':
                self.dpi_dvl = self.dpi_dvl_sqe_loginv
            elif metric == 'sqeuclidean_mod':
                self.dpi_dvl = self.dpi_dvl_sqemod_loginv            
            elif metric == 'cityblock':
                self.dpi_dvl = self.dpi_dvl_city_loginv
            elif metric == 'minkowski':
                self.dpi_dvl = self.dpi_dvl_min_loginv                                
            elif metric == 'minkowski_mod':
                self.dpi_dvl = self.dpi_dvl_minmod_loginv                
            else:
                raise ValueError('Invalid distance definition.')
        elif dist_weights == 'invp':
            self.dist_weight_function = self.dist_weights_invp
            if metric == 'euclidean':
                self.dpi_dvl = self.dpi_dvl_euclid_invp
            elif metric == 'euclidean_mod':
                self.dpi_dvl = self.dpi_dvl_euclidmod_invp                
            elif metric == 'sqeuclidean':
                self.dpi_dvl = self.dpi_dvl_sqe_invp
            elif metric == 'sqeuclidean_mod':
                self.dpi_dvl = self.dpi_dvl_sqemod_invp            
            elif metric == 'cityblock':
                self.dpi_dvl = self.dpi_dvl_city_invp
            elif metric == 'minkowski':
                self.dpi_dvl = self.dpi_dvl_min_invp 
            elif metric == 'minkowski_mod':
                self.dpi_dvl = self.dpi_dvl_minmod_invp                                
            else:
                raise ValueError('Invalid distance definition.')
        elif dist_weights == 'invc':
            self.dist_weight_function = self.dist_weights_invc
            if metric == 'euclidean':
                self.dpi_dvl = self.dpi_dvl_euclid_invc
            elif metric == 'euclidean':
                self.dpi_dvl = self.dpi_dvl_euclid_invc                
            elif metric == 'sqeuclidean':
                self.dpi_dvl = self.dpi_dvl_sqe_invc                
            elif metric == 'sqeuclidean_mod':
                self.dpi_dvl = self.dpi_dvl_sqemod_invc            
            elif metric == 'cityblock':
                self.dpi_dvl = self.dpi_dvl_city_invc
            elif metric == 'minkowski':
                self.dpi_dvl = self.dpi_dvl_min_invc 
            elif metric == 'minkowski_mod':
                self.dpi_dvl = self.dpi_dvl_minmod_invc                                
            else:
                raise ValueError('Invalid distance definition.')
        elif dist_weights == 'gauss':
            self.dist_weight_function = self.dist_weights_gauss
            if metric == 'euclidean':
                self.dpi_dvl = self.dpi_dvl_euclid_gauss
            elif metric == 'euclidean':
                self.dpi_dvl = self.dpi_dvl_euclid_gauss                
            elif metric == 'sqeuclidean':
                self.dpi_dvl = self.dpi_dvl_sqe_gauss                
            elif metric == 'sqeuclidean_mod':
                self.dpi_dvl = self.dpi_dvl_sqemod_gauss            
            elif metric == 'cityblock':
                self.dpi_dvl = self.dpi_dvl_city_gauss
            elif metric == 'minkowski':
                self.dpi_dvl = self.dpi_dvl_min_gauss
            elif metric == 'minkowski_mod':
                self.dpi_dvl = self.dpi_dvl_minmod_gauss                                
            else:
                raise ValueError('Invalid distance definition.')                
        elif dist_weights == 'cosine':
            self.dist_weight_function = self.dist_weights_cosine
            if metric == 'euclidean':
                self.dpi_dvl = self.dpi_dvl_euclid_cosine
            elif metric == 'euclidean':
                self.dpi_dvl = self.dpi_dvl_euclid_cosine                
            elif metric == 'sqeuclidean':
                self.dpi_dvl = self.dpi_dvl_sqe_cosine                
            elif metric == 'sqeuclidean_mod':
                self.dpi_dvl = self.dpi_dvl_sqemod_cosine            
            elif metric == 'cityblock':
                self.dpi_dvl = self.dpi_dvl_city_cosine
            elif metric == 'minkowski':
                self.dpi_dvl = self.dpi_dvl_min_cosine
            elif metric == 'minkowski_mod':
                self.dpi_dvl = self.dpi_dvl_minmod_cosine                                
            else:
                raise ValueError('Invalid distance definition.')                
        elif dist_weights == 'epan':
            self.dist_weight_function = self.dist_weights_epan
            if metric == 'euclidean':
                self.dpi_dvl = self.dpi_dvl_euclid_epan
            elif metric == 'euclidean':
                self.dpi_dvl = self.dpi_dvl_euclid_epan                
            elif metric == 'sqeuclidean':
                self.dpi_dvl = self.dpi_dvl_sqe_epan                
            elif metric == 'sqeuclidean_mod':
                self.dpi_dvl = self.dpi_dvl_sqemod_epan            
            elif metric == 'cityblock':
                self.dpi_dvl = self.dpi_dvl_city_epan
            elif metric == 'minkowski':
                self.dpi_dvl = self.dpi_dvl_min_epan
            elif metric == 'minkowski_mod':
                self.dpi_dvl = self.dpi_dvl_minmod_epan                                
            else:
                raise ValueError('Invalid distance definition.')                
        elif dist_weights == 'tricubic':
            self.dist_weight_function = self.dist_weights_tricubic
            if metric == 'euclidean':
                self.dpi_dvl = self.dpi_dvl_euclid_tricubic
            elif metric == 'euclidean':
                self.dpi_dvl = self.dpi_dvl_euclid_tricubic                
            elif metric == 'sqeuclidean':
                self.dpi_dvl = self.dpi_dvl_sqe_tricubic                
            elif metric == 'sqeuclidean_mod':
                self.dpi_dvl = self.dpi_dvl_sqemod_tricubic            
            elif metric == 'cityblock':
                self.dpi_dvl = self.dpi_dvl_city_tricubic
            elif metric == 'minkowski':
                self.dpi_dvl = self.dpi_dvl_min_tricubic
            elif metric == 'minkowski_mod':
                self.dpi_dvl = self.dpi_dvl_minmod_tricubic                                
            else:
                raise ValueError('Invalid distance definition.')                
        else:
            raise ValueError('Invalid distance weights.') 

        if dist_weights in ['cosine', 'epan', 'tricubic']:
            self.normalize_dists = True
        else:
            self.normalize_dists = False               
                
        self.metric = metric
        self.dist_weights = dist_weights
        
        # miera chyby
        if error_type == 'mae':
            self.error_function = self.compute_mae
            self.dcost_dvl = self.dmae_dvl
        elif error_type == 'mse':
            self.error_function = self.compute_mse
            self.dcost_dvl = self.dmse_dvl
        elif error_type == 'huber':
            self.error_function = self.compute_huber
            self.dcost_dvl = self.dhuber_dvl
        elif error_type == 'cross_entropy':
            self.error_function = self.compute_cre
            self.dcost_dvl = self.dcre_dvl
        else:
            raise ValueError('Invalid error type.')
        self.error_type = error_type
        
        # indexy pre foldy
        self.fold_indices = None
        # cislo aktualneho foldu
        self.current_fold = 0
        # inicializacia nahodneho generatora
        self.random_state = 1
    
    # metoda na inicializaciu vah vysvetlujucich premennych
    # na konkretne vahy,
    # resp. na vahy 1/m (default)
    def init_var_weights(self, xcols, var_weights=None):
        if var_weights is None:
            m_features = len(xcols)
            self.var_weights = np.ones(shape=(m_features))/m_features
        else:
            self.var_weights = np.array(var_weights)
     
    # nahodny vyber podmnoziny pozorovani velkosti batch_size
    # ak je batch_size > 0, nahodne sa vyberie batch_size pozorovani
    # inak cely dataset
    # v stochastic rezime sa pozorovania vyberaju tak, 
    # aby boli rozvrstvene rovnomerne podla hodnoty cielovej premennej
    def choose_subsample(self, X, y, y_class):
        # ak nerobime SGD, vratime vsetky pozorovania 
        if self.batch_size == 0:
            return X, y, list(range(len(X)))
        
        if self.fold_indices is None or self.current_fold >= len(self.fold_indices):
            # vyrobime stratifikovane foldy na vyber pozorovani
            folds = int(len(X)/self.batch_size)
            self.fold_indices = []
            
            if self.stratified_SGD:                    
                skf = StratifiedKFold(y=y_class, n_folds=folds, shuffle=True, random_state = self.random_state)
                for train_index, test_index in skf:
                    self.fold_indices.append(test_index)
            else:
                indices = list(range(len(X)))
                np.random.seed(self.random_state)
                np.random.shuffle(indices)
                for f in range(folds):
                    first = f * self.batch_size
                    last = first + self.batch_size
                    self.fold_indices.append(indices[first:last])

            self.current_fold = 0
            self.random_state += 1

        indices = self.fold_indices[self.current_fold]
        self.current_fold += 1
        X_sample, y_sample = X[indices,:], y[indices]
        return X_sample, y_sample, indices
    
    # hodnotiaca funkcia vzdialenosti e^-c.d
    def dist_weights_exp(self, distance_matrix):
        return math.e ** (-self.c * distance_matrix)
 
    # hodnotiaca funkcia vzdialenosti e^(-d^c)
    def dist_weights_expp(self, distance_matrix):
        return math.e ** (-(distance_matrix**self.c))

    # hodnotiaca funkcia vzdialenosti c^-d
    def dist_weights_ax(self, distance_matrix):
        return self.c ** (-distance_matrix)
    
    # hodnotiaca funkcia vzdialenosti 1/1+d
    def dist_weights_inv(self, distance_matrix):
        return 1/(1+self.c*distance_matrix)
    
    # hodnotiaca funkcia 1/(1+d)^c
    def dist_weights_invc(self, distance_matrix):
        return 1/((1+distance_matrix)**self.c)
    
    # hodnotiaca funkcie vzdialenosti 1/1+d^c
    def dist_weights_invp(self, distance_matrix):
        return 1/(1+distance_matrix**self.c)

    # hodnotiaca funkcia vzdialenosti 1/1+ln(1+d)
    def dist_weights_loginv(self, distance_matrix):
        return 1/(1+np.log(1 + self.c * distance_matrix))
    
    # hodnotiaca funkcia vzdialenosti e^(-(d^2)/2)
    def dist_weights_gauss(self, distance_matrix):
        return math.e ** (-(distance_matrix**2)/2)
    
    # hodnotiaca funkcia kosinusovy kernel
    def dist_weights_cosine(self, distance_matrix):
        return np.cos((math.pi/2) * distance_matrix)  
    
    # hodnotiaca funkcia Epanechnikov kernel
    def dist_weights_epan(self, distance_matrix):
        return 1 - (distance_matrix**2)

    # hodnotiaca funkcia tricubic kernel
    def dist_weights_tricubic(self, distance_matrix):
        return (1-(distance_matrix)**3)**3

    # metoda na vypocet distancnej matice a vektora sum vah susedov 
    # pre jednotlive pozorovania
    def compute_distances(self, X_subsample, X, indices):
        if self.metric == 'cityblock':
            real_weights = self.var_weights
            real_metric = self.metric
            real_p = None
        elif self.metric == 'euclidean':
            real_weights = self.var_weights
            real_metric = self.metric
            real_p = None                        
        elif self.metric == 'sqeuclidean':
            real_weights = self.var_weights
            real_metric = self.metric
            real_p = None
        elif self.metric == 'euclidean_mod':
            real_weights = np.sqrt(np.abs(self.var_weights))
            real_metric = 'euclidean' 
            real_p = None                     
        elif self.metric == 'sqeuclidean_mod':
            real_weights = np.sqrt(np.abs(self.var_weights))
            real_metric = 'sqeuclidean'
            real_p = None
        elif self.metric == 'minkowski':
            real_weights = self.var_weights
            real_metric = 'minkowski'
            real_p = self.p            
        elif self.metric == 'minkowski_mod':
            real_weights = self.var_weights**(1/self.p)
            real_metric = 'minkowski'
            real_p = self.p
            
        # vypocitame Hadamardov sucin
        X_sub_h = np.multiply(X_subsample, real_weights)
        X_h = np.multiply(X, real_weights)
        # vypocet distancnej matice pre nahodne vybrane pozorovania
        # voci vsetkym
        dist_mat = cdist(X_sub_h, X_h, metric=real_metric, p=real_p)
        # odlozime si povodnu distancnu maticu - pre niektore funkcie je potrebna
        
        # niektore hodnotiace funkcie potrebuju d e <0,1>
        if self.normalize_dists:
            max_distance = dist_mat.max()
            dist_mat = dist_mat/max_distance
        
        self.distance_matrix = dist_mat
            
        # aplikacia hodnotiacej funkcie vzdialenosti 
        # na distancnu maticu
        dist_mat = self.dist_weight_function(dist_mat)

        # na diagonalu vlozime nuly, lebo vzdy uvazujeme len susedov okrem daneho pozorovania
        dist_mat[list(range(len(dist_mat))),indices] = 0
                
        # vypocet sum vah susedov pre jednotlive pozorovania
        self.weight_sums = np.sum(dist_mat, axis = 1) 
        self.wdistance_matrix = dist_mat
    
    # naplnenie vektora predikcii
    # predikcia hodnoty cielovej premennej pre kazdy bod datasetu 
    # vypocita sa ako vazeny aritmeticky priemer 
    # hodnot cielovej premennej pre ostatne pozorovania        
    def predict(self, indices, y):
        wdists = self.wdistance_matrix 
        wy = np.multiply(wdists, y)
        wy_sums = np.sum(wy, axis=1) 
        self.predictions = wy_sums/self.weight_sums
        
    # vypocet chyby - MAE (priemer absolutnych hodnot chyb jednotlivych pozorovani)
    def compute_mae(self, y):
        errs = y - self.predictions
        return np.abs(errs).mean() #+ self.lambda_const *((sum(weights)-1)**2)            
    
    # vypocet chyby - MSE (priemer stvorcov chyb jednotlivych pozorovani)
    def compute_mse(self, y):
        errs = y - self.predictions
        return (errs**2).mean() #+ self.lambda_const *((sum(weights)-1)**2)
        
    # vypocet chyby - cross entropy
    def compute_cre(self, y):
        y_pred = self.predictions
        # zakladna verzia, nepouziva sa kvoli problemom pri vypocte logaritmu malych cisel
        #return -(y * np.log(y_pred) + (1-y) * np.log(1-y_pred)).mean()'''        
        errs = np.zeros(shape=len(y))
        for i in range(len(y)):
            if y[i] != 1:
                errs[i] +=  (1-y[i]) * np.log(1-y_pred[i])
            if y[i] != 0:
                errs[i] +=  y[i] * np.log(y_pred[i])
            if np.isnan(errs[i]):
                print('y[i]: ', y[i], 'y_pred[i]: ', y_pred[i])
                raise ValueError()
        return -errs.mean()
        
    # vypocet chyby - Huberova funkcia (priemer)
    def compute_huber(self, y):
        diffs = y - self.predictions
        errs = np.zeros(shape=len(y))
        for i in range(len(y)):
            prediction = self.predictions[i]
            if prediction < y[i] - self.delta:
                errs[i] = self.delta * diffs[i] - 0.5*(self.delta**2)
            elif prediction > y[i] + self.delta:
                errs[i] = -self.delta * diffs[i] - 0.5*(self.delta**2)
            else:
                errs[i] = 0.5*(diffs[i]**2)
        return errs.mean()    

    # derivacia predikovanej hodnoty podla vahy vysvetlujucej premennej v[l]
    # pre euklidovsku vzdialenost
    # pre hodnotiacu funkciu e^-c*d    
    def dpi_dvl_euclid_exp(self, indices, X, y, i, l):
        prediction = self.predictions[i]
        idx = indices[i]
        
        sum_w = self.weight_sums[i]          
        sum_wxp = 0
        for k in range(len(X)):
            if k == idx:
                continue
            w = self.wdistance_matrix[i,k]
            ddik_dvl = self.var_weights[l]*((X[idx,l]-X[k,l]) ** 2)/self.distance_matrix[i,k]
            wxp = w * ddik_dvl * (prediction-y[k])
            sum_wxp += wxp
        return (self.c * self.var_weights[l] * sum_wxp/sum_w)

        
    # derivacia predikovanej hodnoty podla vahy vysvetlujucej premennej v[l]
    # pre stvorec euklidovskej vzdialenosti
    # pre hodnotiacu funkciu e^-c*d    
    def dpi_dvl_sqe_exp(self, indices, X, y, i, l):
        prediction = self.predictions[i]
        idx = indices[i]
        
        sum_w = self.weight_sums[i]          
        sum_wxp = 0
        for k in range(len(X)):
            if k == idx:
                continue
            w = self.wdistance_matrix[i,k]
            wxp = w * ((X[idx,l]-X[k,l]) ** 2) * (prediction-y[k])
            sum_wxp += wxp
        return 2 * self.c * self.var_weights[l] * sum_wxp/sum_w
    
    # derivacia predikovanej hodnoty podla vahy vysvetlujucej premennej v[l]
    # pre euklidovsku vzdialenost
    # modifikacia - vazene druhou odmocninou absolutnej hodnoty vah premennych
    # pre hodnotiacu funkciu e^-c*d    
    def dpi_dvl_euclidmod_exp(self, indices, X, y, i, l):
        prediction = self.predictions[i]
        idx = indices[i]
        
        sum_w = self.weight_sums[i]          
        sum_wxp = 0
        for k in range(len(X)):
            if k == idx:
                continue
            w = self.wdistance_matrix[i,k]
            d = self.distance_matrix[i,k]
            ddik_dvl = ((X[idx,l]-X[k,l])**2)/(2*d)            
            wxp = w * ddik_dvl * (prediction-y[k])
            sum_wxp += wxp
        return self.c * sum_wxp/sum_w
    
    # derivacia predikovanej hodnoty podla vahy vysvetlujucej premennej v[l]
    # pre Minkowskeho vzdialenost
    # pre hodnotiacu funkciu e^-c*d    
    def dpi_dvl_min_exp(self, indices, X, y, i, l):
        prediction = self.predictions[i]
        idx = indices[i]
        
        sum_w = self.weight_sums[i]          
        sum_wxp = 0
        
        if self.vector_computation:
            w = self.wdistance_matrix[i,:]
            
            d = self.distance_matrix[i,:]
            d[idx] = 1
            
            ddik_dvl_c = abs(X[idx*np.ones(len(X),dtype=int),:] - X)**self.p
            ddik_dvl_m = self.p*d**(self.p-1)
            ddik_dvl_m = ddik_dvl_m.reshape(len(X),1)
            ddik_dvl = np.divide(ddik_dvl_c, ddik_dvl_m)
            err = prediction-y[:]
            err = err.reshape(len(X),1)
            wxp = np.multiply(ddik_dvl, err)
            wxp = np.multiply(wxp, w.reshape(len(X),1))
            sum_wxp = np.sum(wxp, axis=0)
        else:            
            for k in range(len(X)):
                if k == idx:
                    continue
                w = self.wdistance_matrix[i,k]
                ddik_dvl = (abs(X[idx,l]-X[k,l])**self.p)/(self.p*(self.distance_matrix[i,k])**(self.p-1))            
                wxp = w * ddik_dvl * (prediction-y[k])
                sum_wxp += wxp
        return self.c * sum_wxp/sum_w * (self.var_weights[l]**(self.p-1))
    
    # derivacia predikovanej hodnoty podla vahy vysvetlujucej premennej v[l]
    # pre Minkowskeho vzdialenost
    # modifikacia - vazene druhou odmocninou absolutnej hodnoty vah premennych
    # pre hodnotiacu funkciu e^-c*d    
    def dpi_dvl_minmod_exp(self, indices, X, y, i, l):
        prediction = self.predictions[i]
        idx = indices[i]
        sum_w = self.weight_sums[i]          
        sum_wxp = 0
        
        if self.vector_computation:
            w = self.wdistance_matrix[i,:]
            
            d = self.distance_matrix[i,:]
            d[idx] = 1
            
            ddik_dvl_c = abs(X[idx*np.ones(len(X),dtype=int),:] - X)**self.p
            ddik_dvl_m = self.p*d**(self.p-1)
            ddik_dvl_m = ddik_dvl_m.reshape(len(X),1)
            ddik_dvl = np.divide(ddik_dvl_c, ddik_dvl_m)
            err = prediction-y[:]
            err = err.reshape(len(X),1)
            wxp = np.multiply(ddik_dvl, err)
            wxp = np.multiply(wxp, w.reshape(len(X),1))
            sum_wxp = np.sum(wxp, axis=0)
        else:
            for k in range(len(X)):
                if k == idx:
                    continue
                w = self.wdistance_matrix[i,k]
                ddik_dvl = (abs(X[idx,l]-X[k,l])**self.p)/(self.p*(self.distance_matrix[i,k])**(self.p-1))            
                wxp = w * ddik_dvl * (prediction-y[k])
                sum_wxp += wxp
                
        return self.c * sum_wxp/sum_w
    

    # derivacia predikovanej hodnoty podla vahy vysvetlujucej premennej v[l]
    # pre stvorec euklidovskej vzdialenosti
    # modifikacia - vazene druhou odmocninou absolutnej hodnoty vah premennych
    # pre hodnotiacu funkciu e^-c*d    
    def dpi_dvl_sqemod_exp(self, indices, X, y, i, l):
        prediction = self.predictions[i]
        idx = indices[i]
        
        sum_w = self.weight_sums[i]          
        sum_wxp = 0
        
        if self.vector_computation:
            w = self.wdistance_matrix[i,:]
            w = w.reshape(len(X),1)
            
            ddik_dvl = (X[idx*np.ones(len(X),dtype=int),:] - X)**2 
            ddik_dvl = np.multiply(ddik_dvl, w)
            err = prediction-y[:]
            err = err.reshape(len(X),1)

            wxp = np.multiply(ddik_dvl, err)
            sum_wxp = np.sum(wxp, axis=0)
        else:        
            for k in range(len(X)):
                if k == idx:
                    continue
                w = self.wdistance_matrix[i,k]
                wxp = w * ((X[idx,l]-X[k,l]) ** 2) * (prediction-y[k])
                sum_wxp += wxp
        return self.c * sum_wxp/sum_w

    # derivacia predikovanej hodnoty podla vahy vysvetlujucej premennej v[l]
    # pre manhattansku vzdialenost
    # pre hodnotiacu funkciu e^-c*d    
    def dpi_dvl_city_exp(self, indices, X, y, i, l):
        prediction = self.predictions[i]
        idx = indices[i]
        
        sum_w = self.weight_sums[i]          
        sum_wxp = 0
        for k in range(len(X)):
            if k == idx:
                continue
            w = self.wdistance_matrix[i,k]
            wxp = w * (abs(X[idx,l]-X[k,l])) * (prediction-y[k])
            sum_wxp += wxp
        return self.c * sum_wxp/sum_w

    # derivacia predikovanej hodnoty podla vahy vysvetlujucej premennej v[l]
    # pre euklidovsku vzdialenost
    # pre hodnotiacu funkciu e^(-d^c)   
    def dpi_dvl_euclid_expp(self, indices, X, y, i, l):
        prediction = self.predictions[i]
        idx = indices[i]
        
        sum_w = self.weight_sums[i]          
        sum_wxp = 0
        for k in range(len(X)):
            if k == idx:
                continue
            w = self.wdistance_matrix[i,k]
            d = self.distance_matrix[i,k]
            dw_dd = w * (d**(self.c-1))             
            ddik_dvl = self.var_weights[l]*((X[idx,l]-X[k,l]) ** 2)/self.distance_matrix[i,k]
            wxp = dw_dd * ddik_dvl * (prediction-y[k])
            sum_wxp += wxp
        return (self.c * self.var_weights[l] * sum_wxp/sum_w)

    # derivacia predikovanej hodnoty podla vahy vysvetlujucej premennej v[l]
    # pre stvorec euklidovskej vzdialenosti
    # pre hodnotiacu funkciu e^(-d^c)   
    def dpi_dvl_sqe_expp(self, indices, X, y, i, l):
        prediction = self.predictions[i]
        idx = indices[i]
        
        sum_w = self.weight_sums[i]          
        sum_wxp = 0
        for k in range(len(X)):
            if k == idx:
                continue
            w = self.wdistance_matrix[i,k]
            d = self.distance_matrix[i,k]
            dw_dd = w * (d**(self.c-1))            
            wxp = dw_dd * ((X[idx,l]-X[k,l]) ** 2) * (prediction-y[k])
            sum_wxp += wxp
        return 2 * self.c * self.var_weights[l] * sum_wxp/sum_w
    
    # derivacia predikovanej hodnoty podla vahy vysvetlujucej premennej v[l]
    # pre euklidovsku vzdialenost
    # modifikacia - vazene druhou odmocninou absolutnej hodnoty vah premennych
    # pre hodnotiacu funkciu e^(-d^c)   
    def dpi_dvl_euclidmod_expp(self, indices, X, y, i, l):
        prediction = self.predictions[i]
        idx = indices[i]
        
        sum_w = self.weight_sums[i]          
        sum_wxp = 0
        for k in range(len(X)):
            if k == idx:
                continue
            w = self.wdistance_matrix[i,k]
            d = self.distance_matrix[i,k]            
            dw_dd = w * (d**(self.c-1))
            ddik_dvl = ((X[idx,l]-X[k,l])**2)/(2*d)            
            wxp = dw_dd * ddik_dvl * (prediction-y[k])
            sum_wxp += wxp
        return self.c * sum_wxp/sum_w
    
    # derivacia predikovanej hodnoty podla vahy vysvetlujucej premennej v[l]
    # pre Minkowskeho vzdialenost
    # pre hodnotiacu funkciu e^(-d^c)   
    def dpi_dvl_min_expp(self, indices, X, y, i, l):
        prediction = self.predictions[i]
        idx = indices[i]
        
        sum_w = self.weight_sums[i]          
        sum_wxp = 0
        for k in range(len(X)):
            if k == idx:
                continue
            w = self.wdistance_matrix[i,k]
            d = self.distance_matrix[i,k]
            dw_dd = w * (d**(self.c-1))
            ddik_dvl = (abs(X[idx,l]-X[k,l])**self.p)/(self.p*(self.distance_matrix[i,k])**(self.p-1))            
            wxp = dw_dd * ddik_dvl * (prediction-y[k])
            sum_wxp += wxp
        return self.c * sum_wxp/sum_w * (self.var_weights[l]**(self.p-1))
    
    # derivacia predikovanej hodnoty podla vahy vysvetlujucej premennej v[l]
    # pre Minkowskeho vzdialenost
    # modifikacia - vazene druhou odmocninou absolutnej hodnoty vah premennych
    # pre hodnotiacu funkciu e^(-d^c)   
    def dpi_dvl_minmod_expp(self, indices, X, y, i, l):
        prediction = self.predictions[i]
        idx = indices[i]
        
        sum_w = self.weight_sums[i]          
        sum_wxp = 0
        for k in range(len(X)):
            if k == idx:
                continue
            w = self.wdistance_matrix[i,k]
            d = self.distance_matrix[i,k]
            dw_dd = w * (d**(self.c-1))
            ddik_dvl = (abs(X[idx,l]-X[k,l])**self.p)/(self.p*(self.distance_matrix[i,k])**(self.p-1))            
            wxp = dw_dd * ddik_dvl * (prediction-y[k])
            sum_wxp += wxp
        return self.c * sum_wxp/sum_w
    

    # derivacia predikovanej hodnoty podla vahy vysvetlujucej premennej v[l]
    # pre stvorec euklidovskej vzdialenosti
    # modifikacia - vazene druhou odmocninou absolutnej hodnoty vah premennych
    # pre hodnotiacu funkciu e^(-d^c)   
    def dpi_dvl_sqemod_expp(self, indices, X, y, i, l):
        prediction = self.predictions[i]
        idx = indices[i]
        
        sum_w = self.weight_sums[i]          
        sum_wxp = 0
        for k in range(len(X)):
            if k == idx:
                continue
            w = self.wdistance_matrix[i,k]
            d = self.distance_matrix[i,k]
            dw_dd = w * (d**(self.c-1))
            wxp = dw_dd * ((X[idx,l]-X[k,l]) ** 2) * (prediction-y[k])
            sum_wxp += wxp
        return self.c * sum_wxp/sum_w

    # derivacia predikovanej hodnoty podla vahy vysvetlujucej premennej v[l]
    # pre manhattansku vzdialenost
    # pre hodnotiacu funkciu e^(-d^c)    
    def dpi_dvl_city_expp(self, indices, X, y, i, l):
        prediction = self.predictions[i]
        idx = indices[i]
        
        sum_w = self.weight_sums[i]          
        sum_wxp = 0
        for k in range(len(X)):
            if k == idx:
                continue
            w = self.wdistance_matrix[i,k]
            d = self.distance_matrix[i,k]
            dw_dd = w * (d**(self.c-1))            
            wxp = dw_dd * (abs(X[idx,l]-X[k,l])) * (prediction-y[k])
            sum_wxp += wxp
        return self.c * sum_wxp/sum_w
    
    # derivacia predikovanej hodnoty podla vahy vysvetlujucej premennej v[l]
    # pre stvorec euklidovskej vzdialenosti
    # pre hodnotiacu funkciu c^-d    
    def dpi_dvl_euclid_ax(self, indices, X, y, i, l):
        prediction = self.predictions[i]
        idx = indices[i]
        
        sum_w = self.weight_sums[i]          
        sum_wxp = 0
        for k in range(len(X)):
            if k == idx:
                continue
            w = self.wdistance_matrix[i,k] * math.log(self.c)            
            ddik_dvl = self.var_weights[l]*((X[idx,l]-X[k,l]) ** 2)/self.distance_matrix[i,k]
            wxp = w * ddik_dvl * (prediction-y[k])
            sum_wxp += wxp
        return self.c * self.var_weights[l] * sum_wxp/sum_w

    # derivacia predikovanej hodnoty podla vahy vysvetlujucej premennej v[l]
    # pre stvorec euklidovskej vzdialenosti
    # pre hodnotiacu funkciu c^-d    
    def dpi_dvl_sqe_ax(self, indices, X, y, i, l):
        prediction = self.predictions[i]
        idx = indices[i]
        
        sum_w = self.weight_sums[i]          
        sum_wxp = 0
        for k in range(len(X)):
            if k == idx:
                continue
            w = self.wdistance_matrix[i,k] * math.log(self.c)
            wxp = w * ((X[idx,l]-X[k,l]) ** 2) * (prediction-y[k])
            sum_wxp += wxp
        return 2 * self.c * self.var_weights[l] * sum_wxp/sum_w
    
    # derivacia predikovanej hodnoty podla vahy vysvetlujucej premennej v[l]
    # pre stvorec euklidovsku vzdialenost
    # modifikacia - vazene druhou odmocninou absolutnej hodnoty vah premennych
    # pre hodnotiacu funkciu c^-d    
    def dpi_dvl_euclidmod_ax(self, indices, X, y, i, l):
        prediction = self.predictions[i]
        idx = indices[i]
        
        sum_w = self.weight_sums[i]          
        sum_wxp = 0
        for k in range(len(X)):
            if k == idx:
                continue
            w = self.wdistance_matrix[i,k] * math.log(self.c)            
            d = self.distance_matrix[i,k]
            ddik_dvl = ((X[idx,l]-X[k,l])**2)/(2*d)            
            wxp = w * ddik_dvl * (prediction-y[k])
            sum_wxp += wxp
        return self.c * sum_wxp/sum_w
    
    # derivacia predikovanej hodnoty podla vahy vysvetlujucej premennej v[l]
    # pre stvorec Minkowskeho vzdialenost
    # pre hodnotiacu funkciu c^-d    
    def dpi_dvl_min_ax(self, indices, X, y, i, l):
        prediction = self.predictions[i]
        idx = indices[i]
        
        sum_w = self.weight_sums[i]          
        sum_wxp = 0
        for k in range(len(X)):
            if k == idx:
                continue
            w = self.wdistance_matrix[i,k] * math.log(self.c)            
            ddik_dvl = (abs(X[idx,l]-X[k,l])**self.p)/(self.p*(self.distance_matrix[i,k])**(self.p-1))            
            wxp = w * ddik_dvl * (prediction-y[k])
            sum_wxp += wxp
        return self.c * sum_wxp/sum_w * (self.var_weights[l]**(self.p-1))

    # derivacia predikovanej hodnoty podla vahy vysvetlujucej premennej v[l]
    # pre stvorec Minkowskeho vzdialenost
    # modifikacia - vazene druhou odmocninou absolutnej hodnoty vah premennych
    # pre hodnotiacu funkciu c^-d    
    def dpi_dvl_minmod_ax(self, indices, X, y, i, l):
        prediction = self.predictions[i]
        idx = indices[i]
        
        sum_w = self.weight_sums[i]          
        sum_wxp = 0
        for k in range(len(X)):
            if k == idx:
                continue
            w = self.wdistance_matrix[i,k] * math.log(self.c)            
            ddik_dvl = (abs(X[idx,l]-X[k,l])**self.p)/(self.p*(self.distance_matrix[i,k])**(self.p-1))            
            wxp = w * ddik_dvl * (prediction-y[k])
            sum_wxp += wxp
        return self.c * sum_wxp/sum_w


    # derivacia predikovanej hodnoty podla vahy vysvetlujucej premennej v[l]
    # pre stvorec euklidovskej vzdialenosti
    # modifikacia - vazene druhou odmocninou absolutnej hodnoty vah premennych
    # pre hodnotiacu funkciu c^-d    
    def dpi_dvl_sqemod_ax(self, indices, X, y, i, l):
        prediction = self.predictions[i]
        idx = indices[i]
        
        sum_w = self.weight_sums[i]          
        sum_wxp = 0
        for k in range(len(X)):
            if k == idx:
                continue
            w = self.wdistance_matrix[i,k] * math.log(self.c)
            wxp = w * ((X[idx,l]-X[k,l]) ** 2) * (prediction-y[k])
            wxp /= ((((X[idx,:]-X[k,:])**self.p).sum())**(1/self.p))            
            sum_wxp += wxp
        return self.c * sum_wxp/sum_w


    # derivacia predikovanej hodnoty podla vahy vysvetlujucej premennej v[l]
    # pre manhattansku vzdialenost
    # pre hodnotiacu funkciu c^-d    
    def dpi_dvl_city_ax(self, indices, X, y, i, l):
        prediction = self.predictions[i]
        idx = indices[i]
        
        sum_w = self.weight_sums[i]          
        sum_wxp = 0
        for k in range(len(X)):
            if k == idx:
                continue
            w = self.wdistance_matrix[i,k] * math.log(self.c)
            wxp = w * (abs(X[idx,l]-X[k,l])) * (prediction-y[k])
            sum_wxp += wxp
        return self.c * sum_wxp/sum_w
        
    # derivacia predikovanej hodnoty podla vahy vysvetlujucej premennej v[l]
    # pre euklidovsku vzdialenost
    # pre hodnotiacu funkciu 1/1+c*d    
    def dpi_dvl_euclid_inv(self, indices, X, y, i, l):
        prediction = self.predictions[i]
        idx = indices[i]
        
        sum_w = self.weight_sums[i]          
        sum_wxp = 0
        for k in range(len(X)):
            if k == idx:
                continue
            w = self.wdistance_matrix[i,k]
            w2 = w ** 2
            ddik_dvl = self.var_weights[l]*((X[idx,l]-X[k,l]) ** 2)/self.distance_matrix[i,k]
            wxp = w2 * ddik_dvl * (prediction-y[k])
            sum_wxp += wxp
        return self.c * self.var_weights[l] * sum_wxp/sum_w
    
    # derivacia predikovanej hodnoty podla vahy vysvetlujucej premennej v[l]
    # pre stvorec euklidovskej vzdialenosti
    # pre hodnotiacu funkciu 1/1+c*d    
    def dpi_dvl_sqe_inv(self, indices, X, y, i, l):
        prediction = self.predictions[i]
        idx = indices[i]
        
        sum_w = self.weight_sums[i]          
        sum_wxp = 0
        for k in range(len(X)):
            if k == idx:
                continue
            w = self.wdistance_matrix[i,k]
            w2 = w ** 2
            wxp = w2 * ((X[idx,l]-X[k,l]) ** 2) * (prediction-y[k])
            sum_wxp += wxp
        return 2 * self.c * self.var_weights[l] * sum_wxp/sum_w
    
    # derivacia predikovanej hodnoty podla vahy vysvetlujucej premennej v[l]
    # euklidovsku vzdialenost
    # modifikacia - vazene druhou odmocninou absolutnej hodnoty vah premennych
    # pre hodnotiacu funkciu 1/1+c*d    
    def dpi_dvl_euclidmod_inv(self, indices, X, y, i, l):
        prediction = self.predictions[i]
        idx = indices[i]
        
        sum_w = self.weight_sums[i]          
        sum_wxp = 0
        for k in range(len(X)):
            if k == idx:
                continue
            w = self.wdistance_matrix[i,k]
            w2 = w ** 2            
            d = self.distance_matrix[i,k]
            ddik_dvl = ((X[idx,l]-X[k,l])**2)/(2*d)            
            wxp = w2 * ddik_dvl * (prediction-y[k])
            sum_wxp += wxp
        return self.c * sum_wxp/sum_w

    # derivacia predikovanej hodnoty podla vahy vysvetlujucej premennej v[l]
    # pre Minkowskeho vzdialenost
    # pre hodnotiacu funkciu 1/1+c*d    
    def dpi_dvl_min_inv(self, indices, X, y, i, l):
        prediction = self.predictions[i]
        idx = indices[i]
        
        sum_w = self.weight_sums[i]          
        sum_wxp = 0
        for k in range(len(X)):
            if k == idx:
                continue
            w = self.wdistance_matrix[i,k]
            w2 = w ** 2
            ddik_dvl = (abs(X[idx,l]-X[k,l])**self.p)/(self.p*(self.distance_matrix[i,k])**(self.p-1))            
            wxp = w2 * ddik_dvl * (prediction-y[k])
            sum_wxp += wxp
        return self.c * sum_wxp/sum_w * (self.var_weights[l]**(self.p-1))

    # derivacia predikovanej hodnoty podla vahy vysvetlujucej premennej v[l]
    # pre Minkowskeho vzdialenost
    # modifikacia - vazene druhou odmocninou absolutnej hodnoty vah premennych
    # pre hodnotiacu funkciu 1/1+c*d    
    def dpi_dvl_minmod_inv(self, indices, X, y, i, l):
        prediction = self.predictions[i]
        idx = indices[i]
        
        sum_w = self.weight_sums[i]          
        sum_wxp = 0
        for k in range(len(X)):
            if k == idx:
                continue
            w = self.wdistance_matrix[i,k]
            w2 = w ** 2
            ddik_dvl = (abs(X[idx,l]-X[k,l])**self.p)/(self.p*(self.distance_matrix[i,k])**(self.p-1))            
            wxp = w2 * ddik_dvl * (prediction-y[k])
            sum_wxp += wxp
        return self.c * sum_wxp/sum_w


    # derivacia predikovanej hodnoty podla vahy vysvetlujucej premennej v[l]
    # pre stvorec euklidovskej vzdialenosti
    # modifikacia - vazene druhou odmocninou absolutnej hodnoty vah premennych
    # pre hodnotiacu funkciu 1/1+c*d    
    def dpi_dvl_sqemod_inv(self, indices, X, y, i, l):
        prediction = self.predictions[i]
        idx = indices[i]
        
        sum_w = self.weight_sums[i]          
        sum_wxp = 0
        for k in range(len(X)):
            if k == idx:
                continue
            w = self.wdistance_matrix[i,k]
            w2 = w ** 2
            wxp = w2 * ((X[idx,l]-X[k,l]) ** 2) * (prediction-y[k])
            sum_wxp += wxp
        return self.c * sum_wxp/sum_w

    # derivacia predikovanej hodnoty podla vahy vysvetlujucej premennej v[l]
    # pre manhattansku vzdialenost
    # pre hodnotiacu funkciu 1/1+c*d
    def dpi_dvl_city_inv(self, indices, X, y, i, l):
        prediction = self.predictions[i]
        idx = indices[i]
        
        sum_w = self.weight_sums[i]          
        sum_wxp = 0
        for k in range(len(X)):
            if k == idx:
                continue
            w = self.wdistance_matrix[i,k]
            w2 = w ** 2
            wxp = w2 * (abs(X[idx,l]-X[k,l])) * (prediction-y[k])
            sum_wxp += wxp
        return self.c * sum_wxp/sum_w
    
    # derivacia predikovanej hodnoty podla vahy vysvetlujucej premennej v[l]
    # pre euklidovskej vzdialenost
    # pre hodnotiacu funkciu 1/1+d^c    
    def dpi_dvl_euclid_invp(self, indices, X, y, i, l):
        prediction = self.predictions[i]
        idx = indices[i]
        
        sum_w = self.weight_sums[i]          
        sum_wxp = 0
        for k in range(len(X)):
            if k == idx:
                continue
            w = self.wdistance_matrix[i,k]
            w2 = w ** 2
            d = self.distance_matrix[i,k]
            dw_dd = w2 * self.c * (d**(self.c-1))            
            ddik_dvl = self.var_weights[l]*((X[idx,l]-X[k,l]) ** 2)/self.distance_matrix[i,k]
            wxp = dw_dd * ddik_dvl * (prediction-y[k])
            sum_wxp += wxp
        return self.var_weights[l] * sum_wxp/sum_w
    

    # derivacia predikovanej hodnoty podla vahy vysvetlujucej premennej v[l]
    # pre stvorec euklidovskej vzdialenosti
    # pre hodnotiacu funkciu 1/1+d^c    
    def dpi_dvl_sqe_invp(self, indices, X, y, i, l):
        prediction = self.predictions[i]
        idx = indices[i]
        
        sum_w = self.weight_sums[i]          
        sum_wxp = 0
        for k in range(len(X)):
            if k == idx:
                continue
            w = self.wdistance_matrix[i,k]
            w2 = w ** 2
            d = self.distance_matrix[i,k]
            dw_dd = w2 * self.c * (d**(self.c-1))
            wxp = dw_dd * ((X[idx,l]-X[k,l]) ** 2) * (prediction-y[k])
            sum_wxp += wxp
        return 2 * self.var_weights[l] * sum_wxp/sum_w
    
    # derivacia predikovanej hodnoty podla vahy vysvetlujucej premennej v[l]
    # pre stvorec euklidovskej vzdialenosti
    # modifikacia - vazene druhou odmocninou absolutnej hodnoty vah premennych
    # pre hodnotiacu funkciu 1/1+d^c   
    def dpi_dvl_euclidmod_invp(self, indices, X, y, i, l):
        prediction = self.predictions[i]
        idx = indices[i]
        
        sum_w = self.weight_sums[i]          
        sum_wxp = 0
        for k in range(len(X)):
            if k == idx:
                continue
            w = self.wdistance_matrix[i,k]
            w2 = w ** 2
            d = self.distance_matrix[i,k]
            dw_dd = w2 * self.c * (d**(self.c-1))
            ddik_dvl = ((X[idx,l]-X[k,l])**2)/(2*d)            
            wxp = dw_dd * ddik_dvl * (prediction-y[k])
            sum_wxp += wxp
        return sum_wxp/sum_w
    
    # derivacia predikovanej hodnoty podla vahy vysvetlujucej premennej v[l]
    # pre Minkowskwho vzdialenosti
    # modifikacia - vazene druhou odmocninou absolutnej hodnoty vah premennych
    # pre hodnotiacu funkciu 1/1+d^c   
    def dpi_dvl_min_invp(self, indices, X, y, i, l):
        prediction = self.predictions[i]
        idx = indices[i]
        
        sum_w = self.weight_sums[i]          
        sum_wxp = 0
        for k in range(len(X)):
            if k == idx:
                continue
            w = self.wdistance_matrix[i,k]
            w2 = w ** 2
            d = self.distance_matrix[i,k]
            dw_dd = w2 * self.c * (d**(self.c-1))            
            ddik_dvl = (abs(X[idx,l]-X[k,l])**self.p)/(self.p*(d**(self.p-1)))           
            wxp = dw_dd * ddik_dvl * (prediction-y[k])
            sum_wxp += wxp
        return sum_wxp/sum_w * (self.var_weights[l]**(self.p-1))

    # derivacia predikovanej hodnoty podla vahy vysvetlujucej premennej v[l]
    # pre Minkowskwho vzdialenosti
    # modifikacia - vazene druhou odmocninou absolutnej hodnoty vah premennych
    # pre hodnotiacu funkciu 1/1+d^c   
    def dpi_dvl_minmod_invp(self, indices, X, y, i, l):
        prediction = self.predictions[i]
        idx = indices[i]
        
        sum_w = self.weight_sums[i]          
        sum_wxp = 0
        for k in range(len(X)):
            if k == idx:
                continue
            w = self.wdistance_matrix[i,k]
            w2 = w ** 2
            d = self.distance_matrix[i,k]
            dw_dd = w2 * self.c * (d**(self.c-1))            
            ddik_dvl = (abs(X[idx,l]-X[k,l])**self.p)/(self.p*(d**(self.p-1)))           
            wxp = dw_dd * ddik_dvl * (prediction-y[k])
            sum_wxp += wxp
        return sum_wxp/sum_w

    # derivacia predikovanej hodnoty podla vahy vysvetlujucej premennej v[l]
    # pre stvorec euklidovskej vzdialenosti
    # modifikacia - vazene druhou odmocninou absolutnej hodnoty vah premennych
    # pre hodnotiacu funkciu 1/1+d^c   
    def dpi_dvl_sqemod_invp(self, indices, X, y, i, l):
        prediction = self.predictions[i]
        idx = indices[i]
        
        sum_w = self.weight_sums[i]          
        sum_wxp = 0
        for k in range(len(X)):
            if k == idx:
                continue
            w = self.wdistance_matrix[i,k]
            w2 = w ** 2
            d = self.distance_matrix[i,k]
            dw_dd = w2 * self.c * (d**(self.c-1))
            wxp = dw_dd * ((X[idx,l]-X[k,l]) ** 2) * (prediction-y[k])
            sum_wxp += wxp
        return sum_wxp/sum_w

    # derivacia predikovanej hodnoty podla vahy vysvetlujucej premennej v[l]
    # pre manhattansku vzdialenost
    # pre hodnotiacu funkciu 1/1+d^c
    def dpi_dvl_city_invp(self, indices, X, y, i, l):
        prediction = self.predictions[i]
        idx = indices[i]
        
        sum_w = self.weight_sums[i]          
        sum_wxp = 0
        for k in range(len(X)):
            if k == idx:
                continue
            w = self.wdistance_matrix[i,k]
            w2 = w ** 2
            d = self.distance_matrix[i,k]
            dw_dd = w2 * self.c * (d**(self.c-1))            
            wxp = dw_dd * (abs(X[idx,l]-X[k,l])) * (prediction-y[k])
            sum_wxp += wxp
        return sum_wxp/sum_w
    
    # derivacia predikovanej hodnoty podla vahy vysvetlujucej premennej v[l]
    # pre euklidovsku vzdialenost
    # pre hodnotiacu funkciu 1/(1+d)^c    
    def dpi_dvl_euclid_invc(self, indices, X, y, i, l):
        prediction = self.predictions[i]
        idx = indices[i]
        
        sum_w = self.weight_sums[i]          
        sum_wxp = 0
        for k in range(len(X)):
            if k == idx:
                continue
            d = self.distance_matrix[i,k]
            dw_dd = self.c / ((1+d)**(self.c+1))
            ddik_dvl = self.var_weights[l]*((X[idx,l]-X[k,l]) ** 2)/self.distance_matrix[i,k]
            wxp = dw_dd * ddik_dvl * (prediction-y[k])
            sum_wxp += wxp
        return 2 * self.var_weights[l] * sum_wxp/sum_w
    

    # derivacia predikovanej hodnoty podla vahy vysvetlujucej premennej v[l]
    # pre stvorec euklidovskej vzdialenosti
    # pre hodnotiacu funkciu 1/(1+d)^c    
    def dpi_dvl_sqe_invc(self, indices, X, y, i, l):
        prediction = self.predictions[i]
        idx = indices[i]
        
        sum_w = self.weight_sums[i]          
        sum_wxp = 0
        for k in range(len(X)):
            if k == idx:
                continue
            d = self.distance_matrix[i,k]
            dw_dd = self.c / ((1+d)**(self.c+1))
            wxp = dw_dd * ((X[idx,l]-X[k,l]) ** 2) * (prediction-y[k])
            sum_wxp += wxp
        return 2 * self.var_weights[l] * sum_wxp/sum_w
    
    # derivacia predikovanej hodnoty podla vahy vysvetlujucej premennej v[l]
    # pre stvorec euklidovskej vzdialenosti
    # modifikacia - vazene druhou odmocninou absolutnej hodnoty vah premennych
    # pre hodnotiacu funkciu 1/(1+d)^c   
    def dpi_dvl_euclidmod_invc(self, indices, X, y, i, l):
        prediction = self.predictions[i]
        idx = indices[i]
        
        sum_w = self.weight_sums[i]          
        sum_wxp = 0
        for k in range(len(X)):
            if k == idx:
                continue
            d = self.distance_matrix[i,k]
            dw_dd = self.c / ((1+d)**(self.c+1))            
            ddik_dvl = ((X[idx,l]-X[k,l])**2)/(2*d)            
            wxp = dw_dd * ddik_dvl * (prediction-y[k])
            sum_wxp += wxp
        return sum_wxp/sum_w
    
    # derivacia predikovanej hodnoty podla vahy vysvetlujucej premennej v[l]
    # pre Minkowskweho vzdialenosti
    # modifikacia - vazene druhou odmocninou absolutnej hodnoty vah premennych
    # pre hodnotiacu funkciu 1/(1+d)^c   
    def dpi_dvl_min_invc(self, indices, X, y, i, l):
        prediction = self.predictions[i]
        idx = indices[i]
        
        sum_w = self.weight_sums[i]          
        sum_wxp = 0
        for k in range(len(X)):
            if k == idx:
                continue
            d = self.distance_matrix[i,k]
            dw_dd = self.c / ((1+d)**(self.c+1))            
            ddik_dvl = (abs(X[idx,l]-X[k,l])**self.p)/(self.p*(d**(self.p-1)))          
            wxp = dw_dd * ddik_dvl * (prediction-y[k])
            sum_wxp += wxp
        return sum_wxp/sum_w * (self.var_weights[l]**(self.p-1))
    
    # derivacia predikovanej hodnoty podla vahy vysvetlujucej premennej v[l]
    # pre Minkowskweho vzdialenosti
    # modifikacia - vazene druhou odmocninou absolutnej hodnoty vah premennych
    # pre hodnotiacu funkciu 1/(1+d)^c   
    def dpi_dvl_minmod_invc(self, indices, X, y, i, l):
        prediction = self.predictions[i]
        idx = indices[i]
        
        sum_w = self.weight_sums[i]          
        sum_wxp = 0
        for k in range(len(X)):
            if k == idx:
                continue
            d = self.distance_matrix[i,k]
            dw_dd = self.c / ((1+d)**(self.c+1))            
            ddik_dvl = (abs(X[idx,l]-X[k,l])**self.p)/(self.p*(d**(self.p-1)))          
            wxp = dw_dd * ddik_dvl * (prediction-y[k])
            sum_wxp += wxp
        return sum_wxp/sum_w
    


    # derivacia predikovanej hodnoty podla vahy vysvetlujucej premennej v[l]
    # pre stvorec euklidovskej vzdialenosti
    # modifikacia - vazene druhou odmocninou absolutnej hodnoty vah premennych
    # pre hodnotiacu funkciu 1/(1+d)^c   
    def dpi_dvl_sqemod_invc(self, indices, X, y, i, l):
        prediction = self.predictions[i]
        idx = indices[i]
        
        sum_w = self.weight_sums[i]          
        sum_wxp = 0
        for k in range(len(X)):
            if k == idx:
                continue
            d = self.distance_matrix[i,k]
            dw_dd = self.c / ((1+d)**(self.c+1))
            wxp = dw_dd * ((X[idx,l]-X[k,l]) ** 2) * (prediction-y[k])
            sum_wxp += wxp
        return sum_wxp/sum_w

    # derivacia predikovanej hodnoty podla vahy vysvetlujucej premennej v[l]
    # pre manhattansku vzdialenost
    # pre hodnotiacu funkciu 1/(1+d)^c
    def dpi_dvl_city_invc(self, indices, X, y, i, l):
        prediction = self.predictions[i]
        idx = indices[i]
        
        sum_w = self.weight_sums[i]          
        sum_wxp = 0
        for k in range(len(X)):
            if k == idx:
                continue
            d = self.distance_matrix[i,k]
            dw_dd = self.c / ((1+d)**(self.c+1))
            wxp = dw_dd * (abs(X[idx,l]-X[k,l])) * (prediction-y[k])
            sum_wxp += wxp
        return sum_wxp/sum_w
    
    # derivacia predikovanej hodnoty podla vahy vysvetlujucej premennej v[l]
    # pre euklidovsku vzdialenost
    # pre hodnotiacu funkciu 1/1+ln(1+d)    
    def dpi_dvl_euclid_loginv(self, indices, X, y, i, l):
        prediction = self.predictions[i]
        idx = indices[i]
        
        sum_w = self.weight_sums[i]          
        sum_wxp = 0
        for k in range(len(X)):
            if k == idx:
                continue
            w = self.wdistance_matrix[i,k]
            w2 = w ** 2
            d = self.distance_matrix[i,k]
            dw_dd = w2 * 1/(1 + self.c * d)             
            ddik_dvl = self.var_weights[l]*((X[idx,l]-X[k,l]) ** 2)/self.distance_matrix[i,k]
            wxp = dw_dd * ddik_dvl * (prediction-y[k])
            sum_wxp += wxp
        return self.var_weights[l] * sum_wxp/sum_w
    

    # derivacia predikovanej hodnoty podla vahy vysvetlujucej premennej v[l]
    # pre stvorec euklidovskej vzdialenosti
    # pre hodnotiacu funkciu 1/1+ln(1+d)    
    def dpi_dvl_sqe_loginv(self, indices, X, y, i, l):
        prediction = self.predictions[i]
        idx = indices[i]
        
        sum_w = self.weight_sums[i]          
        sum_wxp = 0
        for k in range(len(X)):
            if k == idx:
                continue
            w = self.wdistance_matrix[i,k]
            w2 = w ** 2
            d = self.distance_matrix[i,k]
            dw_dd = w2 * 1/(1 + self.c * d)            
            wxp = dw_dd * ((X[idx,l]-X[k,l]) ** 2) * (prediction-y[k])
            sum_wxp += wxp
        return 2 *  self.var_weights[l] * sum_wxp/sum_w
    
    # derivacia predikovanej hodnoty podla vahy vysvetlujucej premennej v[l]
    # pre stvorec euklidovskej vzdialenosti
    # modifikacia - vazene druhou odmocninou absolutnej hodnoty vah premennych
    # pre hodnotiacu funkciu 1/1+ln(1+d)    
    def dpi_dvl_euclidmod_loginv(self, indices, X, y, i, l):
        prediction = self.predictions[i]
        idx = indices[i]
        
        sum_w = self.weight_sums[i]          
        sum_wxp = 0
        for k in range(len(X)):
            if k == idx:
                continue
            w = self.wdistance_matrix[i,k]
            w2 = w ** 2 
            d = self.distance_matrix[i,k]
            dw_dd = w2 * 1/(1 + self.c * d)
            ddik_dvl = ((X[idx,l]-X[k,l])**2)/(2*d)            
            wxp = dw_dd * ddik_dvl * (prediction-y[k])
            sum_wxp += wxp
        return sum_wxp/sum_w
    
    # derivacia predikovanej hodnoty podla vahy vysvetlujucej premennej v[l]
    # pre Minkowskeho vzdialenosti
    # modifikacia - vazene druhou odmocninou absolutnej hodnoty vah premennych
    # pre hodnotiacu funkciu 1/1+ln(1+d)    
    def dpi_dvl_min_loginv(self, indices, X, y, i, l):
        prediction = self.predictions[i]
        idx = indices[i]
        
        sum_w = self.weight_sums[i]          
        sum_wxp = 0
        for k in range(len(X)):
            if k == idx:
                continue
            w = self.wdistance_matrix[i,k]
            w2 = w ** 2 
            d = self.distance_matrix[i,k]
            dw_dd = w2 * 1/(1 + self.c * d)            
            ddik_dvl = (abs(X[idx,l]-X[k,l])**self.p)/(self.p*(d**(self.p-1)))           
            wxp = dw_dd * ddik_dvl * (prediction-y[k])
            sum_wxp += wxp
        return sum_wxp/sum_w * (self.var_weights[l]**(self.p-1))
    
    # derivacia predikovanej hodnoty podla vahy vysvetlujucej premennej v[l]
    # pre Minkowskeho vzdialenosti
    # modifikacia - vazene druhou odmocninou absolutnej hodnoty vah premennych
    # pre hodnotiacu funkciu 1/1+ln(1+d)    
    def dpi_dvl_minmod_loginv(self, indices, X, y, i, l):
        prediction = self.predictions[i]
        idx = indices[i]
        
        sum_w = self.weight_sums[i]          
        sum_wxp = 0
        for k in range(len(X)):
            if k == idx:
                continue
            w = self.wdistance_matrix[i,k]
            w2 = w ** 2 
            d = self.distance_matrix[i,k]
            dw_dd = w2 * 1/(1 + self.c * d)            
            ddik_dvl = (abs(X[idx,l]-X[k,l])**self.p)/(self.p*(d**(self.p-1)))           
            wxp = dw_dd * ddik_dvl * (prediction-y[k])
            sum_wxp += wxp
        return sum_wxp/sum_w
    


    # derivacia predikovanej hodnoty podla vahy vysvetlujucej premennej v[l]
    # pre stvorec euklidovskej vzdialenosti
    # modifikacia - vazene druhou odmocninou absolutnej hodnoty vah premennych
    # pre hodnotiacu funkciu 1/1+ln(1+d)    
    def dpi_dvl_sqemod_loginv(self, indices, X, y, i, l):
        prediction = self.predictions[i]
        idx = indices[i]
        
        sum_w = self.weight_sums[i]          
        sum_wxp = 0
        for k in range(len(X)):
            if k == idx:
                continue
            w = self.wdistance_matrix[i,k]
            w2 = w ** 2 
            d = self.distance_matrix[i,k]
            dw_dd = w2 * 1/(1 + self.c * d)
            wxp = dw_dd * ((X[idx,l]-X[k,l]) ** 2) * (prediction-y[k])
            sum_wxp += wxp
        return sum_wxp/sum_w

    # derivacia predikovanej hodnoty podla vahy vysvetlujucej premennej v[l]
    # pre manhattansku vzdialenost
    # pre hodnotiacu funkciu 1/1+ln(1+d)
    def dpi_dvl_city_loginv(self, indices, X, y, i, l):
        prediction = self.predictions[i]
        idx = indices[i]
        
        sum_w = self.weight_sums[i]          
        sum_wxp = 0
        for k in range(len(X)):
            if k == idx:
                continue
            w = self.wdistance_matrix[i,k]
            w2 = w ** 2
            d = self.distance_matrix[i,k]
            dw_dd = w2 * 1/(1 + self.c * d)            
            wxp = dw_dd * (abs(X[idx,l]-X[k,l])) * (prediction-y[k])
            sum_wxp += wxp
        return sum_wxp/sum_w

    # derivacia predikovanej hodnoty podla vahy vysvetlujucej premennej v[l]
    # pre euklidovsku vzdialenost
    # pre hodnotiacu funkciu e^-(d^2)/2    
    def dpi_dvl_euclid_gauss(self, indices, X, y, i, l):
        prediction = self.predictions[i]
        idx = indices[i]
        
        sum_w = self.weight_sums[i]          
        sum_wxp = 0
        for k in range(len(X)):
            if k == idx:
                continue
            w = self.wdistance_matrix[i,k]
            d = self.distance_matrix[i,k]
            ddik_dvl = self.var_weights[l]*((X[idx,l]-X[k,l]) ** 2)/self.distance_matrix[i,k]
            wxp = w * d * ddik_dvl * (prediction-y[k])
            sum_wxp += wxp
        return (self.var_weights[l] * sum_wxp/sum_w)

        
    # derivacia predikovanej hodnoty podla vahy vysvetlujucej premennej v[l]
    # pre stvorec euklidovskej vzdialenosti
    # pre hodnotiacu funkciu e^-(d^2)/2    
    def dpi_dvl_sqe_gauss(self, indices, X, y, i, l):
        prediction = self.predictions[i]
        idx = indices[i]
        
        sum_w = self.weight_sums[i]          
        sum_wxp = 0
        for k in range(len(X)):
            if k == idx:
                continue
            w = self.wdistance_matrix[i,k]
            d = self.distance_matrix[i,k]
            wxp = w * d * ((X[idx,l]-X[k,l]) ** 2) * (prediction-y[k])
            sum_wxp += wxp
        return 2 * self.var_weights[l] * sum_wxp/sum_w
    
    # derivacia predikovanej hodnoty podla vahy vysvetlujucej premennej v[l]
    # pre euklidovsku vzdialenost
    # modifikacia - vazene druhou odmocninou absolutnej hodnoty vah premennych
    # pre hodnotiacu funkciu e^-(d^2)/2   
    def dpi_dvl_euclidmod_gauss(self, indices, X, y, i, l):
        prediction = self.predictions[i]
        idx = indices[i]
        
        sum_w = self.weight_sums[i]          
        sum_wxp = 0
        for k in range(len(X)):
            if k == idx:
                continue
            w = self.wdistance_matrix[i,k]
            d = self.distance_matrix[i,k]
            ddik_dvl = ((X[idx,l]-X[k,l])**2)/(2*d)            
            wxp = w * d * ddik_dvl * (prediction-y[k])
            sum_wxp += wxp
        return sum_wxp/sum_w
    
    # derivacia predikovanej hodnoty podla vahy vysvetlujucej premennej v[l]
    # pre Minkowskeho vzdialenost
    # pre hodnotiacu funkciu e^-(d^2)/2    
    def dpi_dvl_min_gauss(self, indices, X, y, i, l):
        prediction = self.predictions[i]
        idx = indices[i]
        
        sum_w = self.weight_sums[i]          
        sum_wxp = 0
        for k in range(len(X)):
            if k == idx:
                continue
            w = self.wdistance_matrix[i,k]
            d = self.distance_matrix[i,k]
            ddik_dvl = (abs(X[idx,l]-X[k,l])**self.p)/(self.p*(self.distance_matrix[i,k])**(self.p-1))            
            wxp = w * d * ddik_dvl * (prediction-y[k])
            sum_wxp += wxp
        return sum_wxp/sum_w * (self.var_weights[l]**(self.p-1))
    
    # derivacia predikovanej hodnoty podla vahy vysvetlujucej premennej v[l]
    # pre Minkowskeho vzdialenost
    # modifikacia - vazene druhou odmocninou absolutnej hodnoty vah premennych
    # pre hodnotiacu funkciu e^-(d^2)/2    
    def dpi_dvl_minmod_gauss(self, indices, X, y, i, l):
        prediction = self.predictions[i]
        idx = indices[i]
        
        sum_w = self.weight_sums[i]          
        sum_wxp = 0
        for k in range(len(X)):
            if k == idx:
                continue
            w = self.wdistance_matrix[i,k]
            d = self.distance_matrix[i,k]            
            ddik_dvl = (abs(X[idx,l]-X[k,l])**self.p)/(self.p*(self.distance_matrix[i,k])**(self.p-1))            
            wxp = w * d * ddik_dvl * (prediction-y[k])
            sum_wxp += wxp
        return sum_wxp/sum_w
    

    # derivacia predikovanej hodnoty podla vahy vysvetlujucej premennej v[l]
    # pre stvorec euklidovskej vzdialenosti
    # modifikacia - vazene druhou odmocninou absolutnej hodnoty vah premennych
    # pre hodnotiacu funkciu e^-(d^2)/2    
    def dpi_dvl_sqemod_gauss(self, indices, X, y, i, l):
        prediction = self.predictions[i]
        idx = indices[i]
        
        sum_w = self.weight_sums[i]          
        sum_wxp = 0
        for k in range(len(X)):
            if k == idx:
                continue
            w = self.wdistance_matrix[i,k]
            d = self.distance_matrix[i,k]            
            wxp = w * d * ((X[idx,l]-X[k,l]) ** 2) * (prediction-y[k])
            sum_wxp += wxp
        return sum_wxp/sum_w

    # derivacia predikovanej hodnoty podla vahy vysvetlujucej premennej v[l]
    # pre manhattansku vzdialenost
    # pre hodnotiacu funkciu e^-c*d    
    def dpi_dvl_city_gauss(self, indices, X, y, i, l):
        prediction = self.predictions[i]
        idx = indices[i]
        
        sum_w = self.weight_sums[i]          
        sum_wxp = 0
        for k in range(len(X)):
            if k == idx:
                continue
            w = self.wdistance_matrix[i,k]
            d = self.distance_matrix[i,k]                        
            wxp = w * d * (abs(X[idx,l]-X[k,l])) * (prediction-y[k])
            sum_wxp += wxp
        return sum_wxp/sum_w

    # derivacia predikovanej hodnoty podla vahy vysvetlujucej premennej v[l]
    # pre euklidovsku vzdialenost
    # pre kosinusovy kernel   
    def dpi_dvl_euclid_cosine(self, indices, X, y, i, l):
        prediction = self.predictions[i]
        idx = indices[i]
        
        sum_w = self.weight_sums[i]          
        sum_wxp = 0
        for k in range(len(X)):
            if k == idx:
                continue
            d = self.distance_matrix[i,k]
            dw_dd = math.pi/2 * math.sin(math.pi * d)
            ddik_dvl = self.var_weights[l]*((X[idx,l]-X[k,l]) ** 2)/self.distance_matrix[i,k]
            wxp = dw_dd * ddik_dvl * (prediction-y[k])
            sum_wxp += wxp
        return (self.var_weights[l] * sum_wxp/sum_w)

        
    # derivacia predikovanej hodnoty podla vahy vysvetlujucej premennej v[l]
    # pre stvorec euklidovskej vzdialenosti
    # pre kosinusovy kernel    
    def dpi_dvl_sqe_cosine(self, indices, X, y, i, l):
        prediction = self.predictions[i]
        idx = indices[i]
        
        sum_w = self.weight_sums[i]          
        sum_wxp = 0
        for k in range(len(X)):
            if k == idx:
                continue
            d = self.distance_matrix[i,k]
            dw_dd = math.pi/2 * math.sin(math.pi * d)
            wxp = dw_dd * ((X[idx,l]-X[k,l]) ** 2) * (prediction-y[k])
            sum_wxp += wxp
        return 2 * self.var_weights[l] * sum_wxp/sum_w
    
    # derivacia predikovanej hodnoty podla vahy vysvetlujucej premennej v[l]
    # pre euklidovsku vzdialenost
    # modifikacia - vazene druhou odmocninou absolutnej hodnoty vah premennych
    # pre kosinusovy kernel   
    def dpi_dvl_euclidmod_cosine(self, indices, X, y, i, l):
        prediction = self.predictions[i]
        idx = indices[i]
        
        sum_w = self.weight_sums[i]          
        sum_wxp = 0
        for k in range(len(X)):
            if k == idx:
                continue
            d = self.distance_matrix[i,k]
            dw_dd = math.pi/2 * math.sin(math.pi * d)
            ddik_dvl = ((X[idx,l]-X[k,l])**2)/(2*d)            
            wxp = dw_dd * ddik_dvl * (prediction-y[k])
            sum_wxp += wxp
        return sum_wxp/sum_w
    
    # derivacia predikovanej hodnoty podla vahy vysvetlujucej premennej v[l]
    # pre Minkowskeho vzdialenost
    # pre kosinusovy kernel    
    def dpi_dvl_min_cosine(self, indices, X, y, i, l):
        prediction = self.predictions[i]
        idx = indices[i]
        
        sum_w = self.weight_sums[i]          
        sum_wxp = 0
        for k in range(len(X)):
            if k == idx:
                continue
            d = self.distance_matrix[i,k]
            dw_dd = math.pi/2 * math.sin(math.pi * d)
            ddik_dvl = (abs(X[idx,l]-X[k,l])**self.p)/(self.p*(self.distance_matrix[i,k])**(self.p-1))            
            wxp = dw_dd * ddik_dvl * (prediction-y[k])
            sum_wxp += wxp
        return sum_wxp/sum_w * (self.var_weights[l]**(self.p-1))
    
    # derivacia predikovanej hodnoty podla vahy vysvetlujucej premennej v[l]
    # pre Minkowskeho vzdialenost
    # modifikacia - vazene druhou odmocninou absolutnej hodnoty vah premennych
    # pre kosinusovy kernel    
    def dpi_dvl_minmod_cosine(self, indices, X, y, i, l):
        prediction = self.predictions[i]
        idx = indices[i]
        
        sum_w = self.weight_sums[i]          
        sum_wxp = 0
        for k in range(len(X)):
            if k == idx:
                continue
            d = self.distance_matrix[i,k]
            dw_dd = math.pi/2 * math.sin(math.pi * d)
            ddik_dvl = (abs(X[idx,l]-X[k,l])**self.p)/(self.p*(self.distance_matrix[i,k])**(self.p-1))            
            wxp = dw_dd * ddik_dvl * (prediction-y[k])
            sum_wxp += wxp
        return sum_wxp/sum_w
    

    # derivacia predikovanej hodnoty podla vahy vysvetlujucej premennej v[l]
    # pre stvorec euklidovskej vzdialenosti
    # modifikacia - vazene druhou odmocninou absolutnej hodnoty vah premennych
    # pre kosinusovy kernel    
    def dpi_dvl_sqemod_cosine(self, indices, X, y, i, l):
        prediction = self.predictions[i]
        idx = indices[i]
        
        sum_w = self.weight_sums[i]          
        sum_wxp = 0
        for k in range(len(X)):
            if k == idx:
                continue
            d = self.distance_matrix[i,k]
            dw_dd = math.pi/2 * math.sin(math.pi * d)
            wxp = dw_dd * ((X[idx,l]-X[k,l]) ** 2) * (prediction-y[k])
            sum_wxp += wxp
        return sum_wxp/sum_w

    # derivacia predikovanej hodnoty podla vahy vysvetlujucej premennej v[l]
    # pre manhattansku vzdialenost
    # pre kosinusovy kernel    
    def dpi_dvl_city_cosine(self, indices, X, y, i, l):
        prediction = self.predictions[i]
        idx = indices[i]
        
        sum_w = self.weight_sums[i]          
        sum_wxp = 0
        for k in range(len(X)):
            if k == idx:
                continue
            d = self.distance_matrix[i,k]
            dw_dd = math.pi/2 * math.sin(math.pi * d)
            wxp = dw_dd * (abs(X[idx,l]-X[k,l])) * (prediction-y[k])
            sum_wxp += wxp
        return sum_wxp/sum_w

    # derivacia predikovanej hodnoty podla vahy vysvetlujucej premennej v[l]
    # pre euklidovsku vzdialenost
    # pre Epanechnikov kernel   
    def dpi_dvl_euclid_epan(self, indices, X, y, i, l):
        prediction = self.predictions[i]
        idx = indices[i]
        
        sum_w = self.weight_sums[i]          
        sum_wxp = 0
        for k in range(len(X)):
            if k == idx:
                continue
            d = self.distance_matrix[i,k]
            ddik_dvl = self.var_weights[l]*((X[idx,l]-X[k,l]) ** 2)/self.distance_matrix[i,k]
            wxp = 2 * d * ddik_dvl * (prediction-y[k])
            sum_wxp += wxp
        return (self.var_weights[l] * sum_wxp/sum_w)

        
    # derivacia predikovanej hodnoty podla vahy vysvetlujucej premennej v[l]
    # pre stvorec euklidovskej vzdialenosti
    # pre Epanechnikov kernel    
    def dpi_dvl_sqe_epan(self, indices, X, y, i, l):
        prediction = self.predictions[i]
        idx = indices[i]
        
        sum_w = self.weight_sums[i]          
        sum_wxp = 0
        for k in range(len(X)):
            if k == idx:
                continue
            d = self.distance_matrix[i,k]
            wxp = 2 * d * ((X[idx,l]-X[k,l]) ** 2) * (prediction-y[k])
            sum_wxp += wxp
        return 2 * self.var_weights[l] * sum_wxp/sum_w
    
    # derivacia predikovanej hodnoty podla vahy vysvetlujucej premennej v[l]
    # pre euklidovsku vzdialenost
    # modifikacia - vazene druhou odmocninou absolutnej hodnoty vah premennych
    # pre Epanechnikov kernel   
    def dpi_dvl_euclidmod_epan(self, indices, X, y, i, l):
        prediction = self.predictions[i]
        idx = indices[i]
        
        sum_w = self.weight_sums[i]          
        sum_wxp = 0
        for k in range(len(X)):
            if k == idx:
                continue
            d = self.distance_matrix[i,k]
            ddik_dvl = ((X[idx,l]-X[k,l])**2)/(2*d)            
            wxp = 2 * d * ddik_dvl * (prediction-y[k])
            sum_wxp += wxp
        return sum_wxp/sum_w
    
    # derivacia predikovanej hodnoty podla vahy vysvetlujucej premennej v[l]
    # pre Minkowskeho vzdialenost
    # pre Epanechnikov kernel    
    def dpi_dvl_min_epan(self, indices, X, y, i, l):
        prediction = self.predictions[i]
        idx = indices[i]
        
        sum_w = self.weight_sums[i]          
        sum_wxp = 0
        for k in range(len(X)):
            if k == idx:
                continue
            d = self.distance_matrix[i,k]
            ddik_dvl = (abs(X[idx,l]-X[k,l])**self.p)/(self.p*(self.distance_matrix[i,k])**(self.p-1))            
            wxp = 2 * d * ddik_dvl * (prediction-y[k])
            sum_wxp += wxp
        return sum_wxp/sum_w * (self.var_weights[l]**(self.p-1))
    
    # derivacia predikovanej hodnoty podla vahy vysvetlujucej premennej v[l]
    # pre Minkowskeho vzdialenost
    # modifikacia - vazene druhou odmocninou absolutnej hodnoty vah premennych
    # pre Epanechnikov kernel    
    def dpi_dvl_minmod_epan(self, indices, X, y, i, l):
        prediction = self.predictions[i]
        idx = indices[i]
        
        sum_w = self.weight_sums[i]          
        sum_wxp = 0
        for k in range(len(X)):
            if k == idx:
                continue
            d = self.distance_matrix[i,k]
            ddik_dvl = (abs(X[idx,l]-X[k,l])**self.p)/(self.p*(self.distance_matrix[i,k])**(self.p-1))            
            wxp = 2 * d * ddik_dvl * (prediction-y[k])
            sum_wxp += wxp
        return sum_wxp/sum_w
    

    # derivacia predikovanej hodnoty podla vahy vysvetlujucej premennej v[l]
    # pre stvorec euklidovskej vzdialenosti
    # modifikacia - vazene druhou odmocninou absolutnej hodnoty vah premennych
    # pre Epanechnikov kernel    
    def dpi_dvl_sqemod_epan(self, indices, X, y, i, l):
        prediction = self.predictions[i]
        idx = indices[i]
        
        sum_w = self.weight_sums[i]          
        sum_wxp = 0
        for k in range(len(X)):
            if k == idx:
                continue
            d = self.distance_matrix[i,k]
            wxp = 2 * d * ((X[idx,l]-X[k,l]) ** 2) * (prediction-y[k])
            sum_wxp += wxp
        return sum_wxp/sum_w

    # derivacia predikovanej hodnoty podla vahy vysvetlujucej premennej v[l]
    # pre manhattansku vzdialenost
    # pre Epanechnikov kernel    
    def dpi_dvl_city_epan(self, indices, X, y, i, l):
        prediction = self.predictions[i]
        idx = indices[i]
        
        sum_w = self.weight_sums[i]          
        sum_wxp = 0
        for k in range(len(X)):
            if k == idx:
                continue
            d = self.distance_matrix[i,k]
            wxp = 2 * d * (abs(X[idx,l]-X[k,l])) * (prediction-y[k])
            sum_wxp += wxp
        return sum_wxp/sum_w

    # derivacia predikovanej hodnoty podla vahy vysvetlujucej premennej v[l]
    # pre euklidovsku vzdialenost
    # pre tricubic kernel   
    def dpi_dvl_euclid_tricubic(self, indices, X, y, i, l):
        prediction = self.predictions[i]
        idx = indices[i]
        
        sum_w = self.weight_sums[i]          
        sum_wxp = 0
        for k in range(len(X)):
            if k == idx:
                continue
            d = self.distance_matrix[i,k]
            dw_dd = 3 * (1-d)**2 * (-3 * d**2)
            ddik_dvl = self.var_weights[l]*((X[idx,l]-X[k,l]) ** 2)/self.distance_matrix[i,k]
            wxp = dw_dd * ddik_dvl * (prediction-y[k])
            sum_wxp += wxp
        return (self.var_weights[l] * sum_wxp/sum_w)

        
    # derivacia predikovanej hodnoty podla vahy vysvetlujucej premennej v[l]
    # pre stvorec euklidovskej vzdialenosti
    # pre tricubic kernel    
    def dpi_dvl_sqe_tricubic(self, indices, X, y, i, l):
        prediction = self.predictions[i]
        idx = indices[i]
        
        sum_w = self.weight_sums[i]          
        sum_wxp = 0
        for k in range(len(X)):
            if k == idx:
                continue
            d = self.distance_matrix[i,k]
            dw_dd = 3 * (1-d)**2 * (-3 * d**2)
            wxp = dw_dd * ((X[idx,l]-X[k,l]) ** 2) * (prediction-y[k])
            sum_wxp += wxp
        return 2 * self.var_weights[l] * sum_wxp/sum_w
    
    # derivacia predikovanej hodnoty podla vahy vysvetlujucej premennej v[l]
    # pre euklidovsku vzdialenost
    # modifikacia - vazene druhou odmocninou absolutnej hodnoty vah premennych
    # pre tricubic kernel   
    def dpi_dvl_euclidmod_tricubic(self, indices, X, y, i, l):
        prediction = self.predictions[i]
        idx = indices[i]
        
        sum_w = self.weight_sums[i]          
        sum_wxp = 0
        for k in range(len(X)):
            if k == idx:
                continue
            d = self.distance_matrix[i,k]
            dw_dd = 3 * (1-d)**2 * (-3 * d**2)
            ddik_dvl = ((X[idx,l]-X[k,l])**2)/(2*d)            
            wxp = dw_dd * ddik_dvl * (prediction-y[k])
            sum_wxp += wxp
        return sum_wxp/sum_w
    
    # derivacia predikovanej hodnoty podla vahy vysvetlujucej premennej v[l]
    # pre Minkowskeho vzdialenost
    # pre tricubic kernel    
    def dpi_dvl_min_tricubic(self, indices, X, y, i, l):
        prediction = self.predictions[i]
        idx = indices[i]
        
        sum_w = self.weight_sums[i]          
        sum_wxp = 0
        for k in range(len(X)):
            if k == idx:
                continue
            d = self.distance_matrix[i,k]
            dw_dd = 3 * (1-d)**2 * (-3 * d**2)
            ddik_dvl = (abs(X[idx,l]-X[k,l])**self.p)/(self.p*(self.distance_matrix[i,k])**(self.p-1))            
            wxp = dw_dd * ddik_dvl * (prediction-y[k])
            sum_wxp += wxp
        return sum_wxp/sum_w * (self.var_weights[l]**(self.p-1))
    
    # derivacia predikovanej hodnoty podla vahy vysvetlujucej premennej v[l]
    # pre Minkowskeho vzdialenost
    # modifikacia - vazene druhou odmocninou absolutnej hodnoty vah premennych
    # pre tricubic kernel    
    def dpi_dvl_minmod_tricubic(self, indices, X, y, i, l):
        prediction = self.predictions[i]
        idx = indices[i]
        
        sum_w = self.weight_sums[i]          
        sum_wxp = 0
        for k in range(len(X)):
            if k == idx:
                continue
            d = self.distance_matrix[i,k]
            dw_dd = 3 * (1-d)**2 * (-3 * d**2)
            ddik_dvl = (abs(X[idx,l]-X[k,l])**self.p)/(self.p*(self.distance_matrix[i,k])**(self.p-1))            
            wxp = dw_dd * ddik_dvl * (prediction-y[k])
            sum_wxp += wxp
        return sum_wxp/sum_w
    

    # derivacia predikovanej hodnoty podla vahy vysvetlujucej premennej v[l]
    # pre stvorec euklidovskej vzdialenosti
    # modifikacia - vazene druhou odmocninou absolutnej hodnoty vah premennych
    # pre tricubic kernel    
    def dpi_dvl_sqemod_tricubic(self, indices, X, y, i, l):
        prediction = self.predictions[i]
        idx = indices[i]
        
        sum_w = self.weight_sums[i]          
        sum_wxp = 0
        for k in range(len(X)):
            if k == idx:
                continue
            d = self.distance_matrix[i,k]
            dw_dd = 3 * (1-d)**2 * (-3 * d**2)
            wxp = dw_dd * ((X[idx,l]-X[k,l]) ** 2) * (prediction-y[k])
            sum_wxp += wxp
        return sum_wxp/sum_w

    # derivacia predikovanej hodnoty podla vahy vysvetlujucej premennej v[l]
    # pre manhattansku vzdialenost
    # pre tricubic kernel    
    def dpi_dvl_city_tricubic(self, indices, X, y, i, l):
        prediction = self.predictions[i]
        idx = indices[i]
        
        sum_w = self.weight_sums[i]          
        sum_wxp = 0
        for k in range(len(X)):
            if k == idx:
                continue
            d = self.distance_matrix[i,k]
            dw_dd = 3 * (1-d)**2 * (-3 * d**2)
            wxp = dw_dd * (abs(X[idx,l]-X[k,l])) * (prediction-y[k])
            sum_wxp += wxp
        return sum_wxp/sum_w
    
    # derivacia regularizacie
    def dreg_dvl(self, l):
        vl = self.var_weights[l]        
        if self.reg_method == 'L1':
            return np.sign(vl)
        elif self.reg_method == 'L0':
            return 2 * (1e+2/((1+math.e**(-1e+2 * vl))**2))*(math.e**(-1e+2 * vl))
                
    # derivacia ucelovej funkcie (MAE + lambda * regularizacia) podla vahy vysvetlujucej premennej v[l]
    def dmae_dvl(self, indices, X, y, l):
        n_samples = len(indices)
        sum_dpi = 0
        for i in range(n_samples):
            idx = indices[i]
            sum_dpi += np.sign(y[idx]-self.predictions[i]) * self.dpi_dvl(indices, X, y, i, l)                       
            
        dreg_dvl = self.dreg_dvl(l)
        return -sum_dpi/n_samples + self.lambda_const * dreg_dvl 

    # derivacia ucelovej funkcie (MSE + lambda * regularizacia) podla vahy vysvetlujucej premennej v[l]
    def dmse_dvl(self, indices, X, y, l):
        n_samples = len(indices)
        sum_dpi = 0
        for i in range(n_samples):
            dpi = self.dpi_dvl(indices, X, y, i, l)
            prediction = self.predictions[i]
            diff = prediction - y[indices[i]]                
            sum_dpi += 2 * diff * dpi

        dreg_dvl = self.dreg_dvl(l)
        return sum_dpi/n_samples + self.lambda_const * dreg_dvl 
    
    # derivacia ucelovej funkcie (Huber + lambda * regularizacia) podla vahy vysvetlujucej premennej v[l]
    def dhuber_dvl(self, indices, X, y, l):
        n_samples = len(indices)
        sum_dpi = 0
        for i in range(n_samples):
            if self.predictions[i] < y[indices[i]] - self.delta:
                dpi = -self.delta * self.dpi_dvl(indices, X, y, i, l)
            elif self.predictions[i] > y[indices[i]] + self.delta:
                dpi = self.delta * self.dpi_dvl(indices, X, y, i, l)
            else:
                dpi = (self.predictions[i] - y[indices[i]]) * self.dpi_dvl(indices, X, y, i, l)
            sum_dpi += dpi

        dreg_dvl = self.dreg_dvl(l)
        return sum_dpi/n_samples + self.lambda_const * dreg_dvl 
    
    # derivacia ucelovej funkcie (cross entropy + lambda * regularizacia) podla vahy vysvetlujucej premennej v[l]
    def dcre_dvl(self, indices, X, y, l):
        y_pred = self.predictions
        n_samples = len(indices)
        sum_dpi = 0
        for i in range(n_samples):
            dpi = self.dpi_dvl(indices, X, y, i, l)
            #sum_dpi += (y[indices[i]] * (1/y_pred[i]) - (1-y[indices[i]]) * (1/(1-y_pred[i]))) * dpi
            if y[indices[i]] != 1:
                sum_dpi -=  (1-y[indices[i]]) * (1/(1-y_pred[i])) * dpi
            if y[indices[i]] != 0:
                sum_dpi +=  y[indices[i]] * (1/y_pred[i]) * dpi
            
        dreg_dvl = self.dreg_dvl(l)
        return -sum_dpi/n_samples + self.lambda_const * dreg_dvl 
    
        
    # pomocna metoda na upravu vah na nezaporne hodnoty
    # najdeme zapornu vahu s najvacsou absolutnou hodnotou a jej index
    # podla toho vypocitame nove eta ako podiel tejto vahy a prislusnej zlozky gradientu
    def correct_var_weights(self, old_var_weights, gradient_vector):
        min_weight = self.var_weights.min()
        min_weight_index = 0
        if min_weight >= 0:
            return
        
        use_new_eta = False
        # 1. moznost - zaporne vahy jednoducho vynulujeme
        if self.check_neg_var_weights == 'zero':
            tmp_var_weights = np.maximum(self.var_weights,np.zeros(shape=len(self.var_weights)))                                    
            if tmp_var_weights.max() == 0:
                use_new_eta = True
            else:
                self.var_weights = tmp_var_weights
        # 2. moznost - zaporne vahy dame do absolutnych hodnot                                
        elif self.check_neg_var_weights == 'abs':
            self.var_weights = abs(self.var_weights)
        # 3. moznost - prepocitame eta    
        if self.check_neg_var_weights == 'new_eta' or use_new_eta:
           new_eta = self.eta
           for idx in range(len(self.var_weights)):
               if self.var_weights[idx] >= 0:
                   continue
               new_eta_candidate = old_var_weights[idx] / gradient_vector[idx]                        
               if new_eta_candidate < new_eta:
                   new_eta = new_eta_candidate
                   min_weight_index = idx                
           self.var_weights = old_var_weights - gradient_vector * new_eta
        
        for idx in range(len(self.var_weights)):
            if abs(self.var_weights[idx]) < self.prec:
                self.var_weights[idx] = 0
                                
        if self.debug and self.check_neg_var_weights == 'new_eta':
            min_weight_check = self.var_weights.min()
            self.f.write('\nMin weight:' + str(min_weight))
            self.f.write('\nNew eta:' + str(new_eta))
            self.f.write('\nIndex:' + str(min_weight_index))
            self.f.write('\nOriginal weight:' + str(old_var_weights[min_weight_index]))
            self.f.write('\nOriginal gradient:' + str(gradient_vector[min_weight_index]))
            self.f.write('\nNew min. weight:' + str(min_weight_check))

    # aktualizacia vah vysvetlujucich premennych
    # vypocet novych vah zo starych podla vzorca 
    # nova vaha = stara vaha - eta * gradient
    def update_var_weights(self, t, gradient_vector):   
        self.var_weights = self.var_weights - gradient_vector * self.eta        

    # pomocna metoda na vypis parametrov na zaciatok suboru            
    def print_params(self):
        f = self.f
        m = self.metric
        if self.metric in ['minkowski','minkowski_mod']:
            m = m + '(p = ' + str(self.p) + ')'
        f.write('\nDistance definition: ' + m) 
        f.write('\nDistance constant: ' + str(self.c))
        f.write('\nDistance weight function: ' + self.dist_weights)
        f.write('\nLambda: ' + str(self.lambda_const))
        f.write('\nEta: ' + str(self.eta))
        mode = '\nMode: '
        if self.batch_size <= 0:
            mode += 'non-stochastic'
        else:
            mode += 'stochastic'
            mode += ' (batchsize' + str(self.batch_size) + ')'
        f.write(mode)
        f.write('\nItercount: ' + str(self.n_iters))
        f.write('\nEpsilon: ' + str(self.eps))
        f.write('\nPrecision: ' + str(self.prec))
        f.write('\nError type: ' + self.error_type)
        if self.error_type == 'huber':
            f.write('\nDelta: ' + str(self.delta))
        if self.normalize_grad:
            flag = 'enabled'
        else:
            flag = 'disabled'
        f.write('\nGradient normalization: ' + flag)
        if self.check_neg_var_weights != None:
            flag = self.check_neg_var_weights
        else:
            flag = 'disabled'
        f.write('\nNegative weight checking: ' + flag)

    # samotna metoda pre vyber premennych
    def select_vars(self, X, y, xcols, maxvars=None, init_var_weights=None):
 
		# krok 1 - standardizacia - data su uz standardizovane inym programom
		# krok 2 - nastavenie uvodnych vah
        m_features = len(xcols)
        self.init_var_weights(xcols, init_var_weights)
        
        if self.debug:
            f = self.f
            self.print_params()
            f.write('\nDataset size: ' + str(len(X)) + ' samples')
            f.write('\nDataset dimensionality: ' + str(m_features) + ' features')

        # najlepsia chyba
        opt_error = None
        
        # vypocitame triedy pre pozorovania podla percentilu
        # pouziva sa pre vytvorenie stratifikovanych vzoriek
        if len(np.unique(y)) > 10:
            p = np.arange(1,10) * 10
            bins = np.percentile(y,p)
            y_class = np.digitize(y,bins)
        else:
            y_class = y
        
        # krok 3 - hlavna slucka
        for t in range(1,self.n_iters+1):
            if self.debug:
                start_time = time.time()
                f.write('\n\nIteration:' + str(t))
                print('\nIteration:' + str(t))                
                f.write('\nVariable weights:' +  str(self.var_weights.tolist()))
                sum_vk = abs(self.var_weights).sum()                
                f.write('\nSum |vk|:' + str(sum_vk))
              
            # krok 3a - nahodny vyber pozorovani
            X_sample, y_sample, indices = self.choose_subsample(X,y,y_class)
                        
            # krok 3b - vypocet matice vzdialenosti
            # krok 3c - vypocet matice ohodnotenych vzdialenosti
            self.compute_distances(X_sample, X, indices)
            min_sum = self.weight_sums.min() 
            if min_sum < self.min_weight_sum:
                if self.debug:
                    f.write('\nStopped by stopping condition (min. weight sum) at iteration:' + str(t))
                    print('\nStopped by stopping condition (min. weight sum) at iteration:' + str(t))
                break
                
            if self.debug:
                print('Min. weight sum:', min_sum) 
                self.f.write('\nMin. weight sum:' + str(min_sum))                
                end_time = time.time()

			# krok 3d - vypocet vektora predikcii
            self.predict(indices, y)
            if self.debug:
                end_time = time.time()

            # krok 3e - vypocet hodnoty priemernej chyby
            new_error = self.error_function(y_sample)
            if opt_error == None or new_error < opt_error:
                opt_error = new_error
            
            if self.debug:
                if self.reg_method == 'L1':
                    regularization = abs(self.var_weights).sum()
                elif self.reg_method == 'L0':
                    regularization = 2*(1/(1 + math.e**(-100 * self.var_weights)) - 0.5).sum()                    
                f.write('\nRegularizer:' + str(regularization))
                print('Regularizer:' + str(regularization))
                cost = new_error + self.lambda_const * regularization
                f.write('\nError:' + str(new_error) + ' Cost function: ' + str(cost))
                print('Error:' + str(new_error) + ' Cost function: ' + str(cost))
                end_time = time.time()
            
            # krok 3f - vypocet gradientu
            if self.vector_computation:
                var_nrs = list(range(m_features))
                gradient_vector = self.dcost_dvl(indices, X, y, var_nrs)
                bidx = np.logical_and(gradient_vector > 0,self.var_weights == 0)
                gradient_vector[bidx] = 0
            else:               
                gradient_vector = np.zeros(shape=(m_features))            
                for l in range(m_features):
                    gradient =  self.dcost_dvl(indices, X, y, l)
                    gradient_vector[l] = gradient
                    # pre nulove vahy musi byt gradient zaporny               
                    if self.var_weights[l] == 0 and gradient_vector[l] > 0:
                        gradient_vector[l] = 0

            if self.debug:
                end_time = time.time()
                
            # krok 3g - normalizacia gradientu - volitelne
            norm_gradient = abs(gradient_vector).sum()            
            if self.normalize_grad:
                gradient_vector = gradient_vector/norm_gradient                
            if self.debug:
                f.write('\nGradient:' + str(gradient_vector.tolist()))
            
            # krok 3h - aktualizacia vah
            tmp_var_weights = self.var_weights
            self.update_var_weights(t, gradient_vector)
			
            # krok 3i - zistovanie pritomnosti zapornej vahy premennej
            if self.check_neg_var_weights != None:
                self.correct_var_weights(tmp_var_weights, gradient_vector)
                
            if self.debug:
                print('Min. var. weight:', self.var_weights.min())

            # stop test - ak norma gradientu je mensia ako epsilon, skoncime
            if norm_gradient < self.eps:
                if self.debug:
                    f.write('\nStopped by stopping condition (gradient) at iteration:' + str(t))
                    print('\nStopped by stopping condition (gradient) at iteration:' + str(t))
                break
            
            # vypisanie zoznamu prvych 100 premennych po kazdych 100 iteraciach
            if self.debug and self.detail:
                if t % 100 == 0:
                    var_list = zip(xcols,tmp_var_weights)
                    var_list = sorted(var_list,key=lambda var_list: var_list[1],reverse=True)
                    if m_features > 100:
                        var_list = var_list[:100]
                    for var in var_list:
                        f.write('\nVariable: ' + str(var[0]) + ' with weight: ' + str(var[1]))
                                    
            if self.debug:
                end_time = time.time()
                f.write('\nIteration time: ' + str(end_time - start_time))
                f.flush()
                
        w_list = zip(xcols,range(len(xcols)),tmp_var_weights)
        w_list = sorted(w_list,key=lambda w_list: w_list[2],reverse=True)
        # na konci vypiseme do suboru finalne vybrane premenne s ich vahami                
        if self.debug:
            f.write('\n\nFinal error:  ' + str(new_error))
            f.write('\n\nFinal variable weights:')
            for var in w_list:
                f.write('\nVariable: ' + str(var[0]) + ' with weight: ' + str(var[2]))
            f.write('\n')
            
        res_colnames = [v[0] for v in w_list] 
        res_colnrs = [v[1] for v in w_list]
        res_weights = [w[2] for w in w_list]
        # vratime zoznam premennych zoradeny podla vah a prislusne ich finalne vahy        
        return res_colnames, res_colnrs, res_weights, new_error
    
    # pomocna metoda - otvorenie suboru pre logovanie so specialnym nazvom
    def open_output_file(self, output_folder, input_folder, X_file, y_file):
        filename = X_file + '_l' + str(self.lambda_const) + '_e' + str(self.eta) +'_it' + str(self.n_iters) 
        filename += '_' + self.error_type + '_' + self.metric + '_' + self.dist_weights + '_' 
        if self.metric in ['minkowski','minkowski_mod']:
            filename += 'p' + str(self.p)
        filename  +=  '.txt'
        self.f = open(output_folder + '/' + filename,'w')
        self.f.write('\nInput folder: ' + input_folder)
        self.f.write('\nInput file for X: ' + X_file)        
        self.f.write('\nInput file for y: ' + y_file)  
                            
    