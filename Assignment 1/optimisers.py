def updateWeightsSGD(m, params,gradients,layers, learning_rate, weight_decay):
    for i in range(1,len(layers)):
        params["W"+str(i)]-= learning_rate*gradients["dw"+str(i)]
        params["b"+str(i)]-= learning_rate*gradients["db"+str(i)]
        params["W"+str(i)]-= learning_rate*(weight_decay/m)*params["W"+str(i)]
        params["b"+str(i)]-= learning_rate*(weight_decay/m)*params["b"+str(i)]
    return(params)


def updateWeightsMomentum(m, params, gradients, M, layers, learning_rate,weight_decay, beta):
    for i in range(1,len(layers)):
        M["W"+str(i)]=beta*M["W"+str(i)]+(1-beta)*gradients["dw"+str(i)]
        M["b"+str(i)]=beta*M["b"+str(i)]+(1-beta)*gradients["db"+str(i)]
        params["W"+str(i)]-= learning_rate* M["W"+str(i)]
        params["b"+str(i)]-= learning_rate*M["b"+str(i)]
        params["W"+str(i)]-= learning_rate*(weight_decay/m)*params["W"+str(i)]
        params["b"+str(i)]-= learning_rate*(weight_decay/m)*params["b"+str(i)]
    return(params,M)


def updateWeightsRMS(m, params, gradients,R, layers, learning_rate, weight_decay, beta ,eps):

    for i in range(1,len(layers)):
        R["W"+str(i)]=beta*R["W"+str(i)]+(1-beta)*np.power(gradients["dw"+str(i)],2)
        R["b"+str(i)]=beta*R["b"+str(i)]+(1-beta)*np.power(gradients["db"+str(i)],2)
        params["W"+str(i)]-= (learning_rate*gradients["dw"+str(i)])/(np.sqrt(R["W"+str(i)])+eps)
        params["b"+str(i)]-= (learning_rate*gradients["db"+str(i)])/(np.sqrt(R["b"+str(i)])+eps)
        params["W"+str(i)]-= learning_rate*(weight_decay/m)*params["W"+str(i)]
        params["b"+str(i)]-= learning_rate*(weight_decay/m)*params["b"+str(i)]
    return(params,R)


def updateWeightsNesterov(m, params, lookahead_grads, M, layers, learning_rate, weight_decay, beta):
    
    for i in range(1,len(layers)):
        M["W"+str(i)]=beta*M["W"+str(i)]+(1-beta)*lookahead_grads["dw"+str(i)]
        M["b"+str(i)]=beta*M["b"+str(i)]+(1-beta)*lookahead_grads["db"+str(i)]
        params["W"+str(i)]-= learning_rate* M["W"+str(i)]
        params["b"+str(i)]-= learning_rate*M["b"+str(i)]
        params["W"+str(i)]-= learning_rate*(weight_decay/m)*params["W"+str(i)]
        params["b"+str(i)]-= learning_rate*(weight_decay/m)*params["b"+str(i)]
    return(params,M)

def updateWeightsNAdam(m, params, lookahead_grads, layers, M, R, learning_rate, weight_decay, gamma1, gamma2, eps, t):
    M_c={}
    R_c={}
    for i in range(1,len(layers)):
        M["W"+str(i)]=gamma1*M["W"+str(i)]+(1-gamma1)*lookahead_grads["dw"+str(i)]
        M["b"+str(i)]=gamma1*M["b"+str(i)]+(1-gamma1)*lookahead_grads["db"+str(i)]
        M_c["W"+str(i)]=M["W"+str(i)]/(1-np.power(gamma1,t))
        M_c["b"+str(i)]=M["b"+str(i)]/(1-np.power(gamma1,t))
        R["W"+str(i)]=gamma2*R["W"+str(i)]+(1-gamma2)*np.power(lookahead_grads["dw"+str(i)],2)
        R["b"+str(i)]=gamma2*R["b"+str(i)]+(1-gamma2)*np.power(lookahead_grads["db"+str(i)],2)
        R_c["W"+str(i)]=R["W"+str(i)]/(1-np.power(gamma2,t))
        R_c["b"+str(i)]=R["b"+str(i)]/(1-np.power(gamma2,t))
        params["W"+str(i)]-= (learning_rate*M_c["W"+str(i)])/(np.sqrt(R_c["W"+str(i)])+eps)
        params["b"+str(i)]-= (learning_rate*M_c["b"+str(i)])/(np.sqrt(R_c["b"+str(i)])+eps)
        params["W"+str(i)]-= learning_rate*(weight_decay/m)*params["W"+str(i)]
        params["b"+str(i)]-= learning_rate*(weight_decay/m)*params["b"+str(i)]
        
    return(params,M,R)

def updateWeightsAdam(m, params, gradients, layers, M, R, learning_rate, weight_decay, gamma1, gamma2, eps, t):
    M_c={}
    R_c={}
    for i in range(1,len(layers)):
        M["W"+str(i)]=gamma1*M["W"+str(i)]+(1-gamma1)*gradients["dw"+str(i)]
        M["b"+str(i)]=gamma1*M["b"+str(i)]+(1-gamma1)*gradients["db"+str(i)]
        M_c["W"+str(i)]=M["W"+str(i)]/(1-np.power(gamma1,t))
        M_c["b"+str(i)]=M["b"+str(i)]/(1-np.power(gamma1,t))
        R["W"+str(i)]=gamma2*R["W"+str(i)]+(1-gamma2)*np.power(gradients["dw"+str(i)],2)
        R["b"+str(i)]=gamma2*R["b"+str(i)]+(1-gamma2)*np.power(gradients["db"+str(i)],2)
        R_c["W"+str(i)]=R["W"+str(i)]/(1-np.power(gamma2,t))
        R_c["b"+str(i)]=R["b"+str(i)]/(1-np.power(gamma2,t))
        params["W"+str(i)]-= (learning_rate*M_c["W"+str(i)])/(np.sqrt(R_c["W"+str(i)])+eps)
        params["b"+str(i)]-= (learning_rate*M_c["b"+str(i)])/(np.sqrt(R_c["b"+str(i)])+eps)
        params["W"+str(i)]-= learning_rate*(weight_decay/m)*params["W"+str(i)]
        params["b"+str(i)]-= learning_rate*(weight_decay/m)*params["b"+str(i)]
    return(params,M,R)
