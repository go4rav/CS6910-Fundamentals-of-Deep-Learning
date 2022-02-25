
# Updates weights through Stochastic Gradient Descent
def UpdateWeightsSGD(m, params,gradients,layers, learning_rate, weight_decay):
    for i in range(1,len(layers)):
        params["W"+str(i)]-= learning_rate*gradients["dw"+str(i)]
        params["b"+str(i)]-= learning_rate*gradients["db"+str(i)]
        params["W"+str(i)]-= learning_rate*(weight_decay/m)*params["W"+str(i)]
        params["b"+str(i)]-= learning_rate*(weight_decay/m)*params["b"+str(i)]
    return(params)


# Updates weights through Momentum Gradient Descent
def UpdateWeightsMomentum(m, params, gradients, M, layers, learning_rate,weight_decay, beta1):
    for i in range(1,len(layers)):
        M["W"+str(i)]=beta1*M["W"+str(i)]+(1-beta1)*gradients["dw"+str(i)]
        M["b"+str(i)]=beta1*M["b"+str(i)]+(1-beta1)*gradients["db"+str(i)]
        params["W"+str(i)]-= learning_rate* M["W"+str(i)]
        params["b"+str(i)]-= learning_rate*M["b"+str(i)]
        params["W"+str(i)]-= learning_rate*(weight_decay/m)*params["W"+str(i)]
        params["b"+str(i)]-= learning_rate*(weight_decay/m)*params["b"+str(i)]
    return(params,M)


# Updates weights through RMSprop Gradient Descent
def UpdateWeightsRMS(m, params, gradients,R, layers, learning_rate, weight_decay, beta1 ,eps):

    for i in range(1,len(layers)):
        R["W"+str(i)]=beta1*R["W"+str(i)]+(1-beta1)*np.power(gradients["dw"+str(i)],2)
        R["b"+str(i)]=beta1*R["b"+str(i)]+(1-beta1)*np.power(gradients["db"+str(i)],2)
        params["W"+str(i)]-= (learning_rate*gradients["dw"+str(i)])/(np.sqrt(R["W"+str(i)])+eps)
        params["b"+str(i)]-= (learning_rate*gradients["db"+str(i)])/(np.sqrt(R["b"+str(i)])+eps)
        params["W"+str(i)]-= learning_rate*(weight_decay/m)*params["W"+str(i)]
        params["b"+str(i)]-= learning_rate*(weight_decay/m)*params["b"+str(i)]
    return(params,R)


# Updates weights through Nesterov Accelerated Gradient Descent
def UpdateWeightsNesterov(m, params, lookahead_grads, M, layers, learning_rate, weight_decay, beta1):
    
    for i in range(1,len(layers)):
        M["W"+str(i)]=beta1*M["W"+str(i)]+(1-beta1)*lookahead_grads["dw"+str(i)]
        M["b"+str(i)]=beta1*M["b"+str(i)]+(1-beta1)*lookahead_grads["db"+str(i)]
        params["W"+str(i)]-= learning_rate* M["W"+str(i)]
        params["b"+str(i)]-= learning_rate*M["b"+str(i)]
        params["W"+str(i)]-= learning_rate*(weight_decay/m)*params["W"+str(i)]
        params["b"+str(i)]-= learning_rate*(weight_decay/m)*params["b"+str(i)]
    return(params,M)

# Updates weights through Nesterov Adam Gradient Descent
def UpdateWeightsNAdam(m, params, lookahead_grads, layers, M, R, learning_rate, weight_decay, beta1, beta2, eps, t):
    M_c={}
    R_c={}
    for i in range(1,len(layers)):
        M["W"+str(i)]=beta1*M["W"+str(i)]+(1-beta1)*lookahead_grads["dw"+str(i)]
        M["b"+str(i)]=beta1*M["b"+str(i)]+(1-beta1)*lookahead_grads["db"+str(i)]

        M_c["W"+str(i)]=M["W"+str(i)]/(1-np.power(beta1,t))   # bias correction
        M_c["b"+str(i)]=M["b"+str(i)]/(1-np.power(beta1,t))
        R["W"+str(i)]=beta2*R["W"+str(i)]+(1-beta2)*np.power(lookahead_grads["dw"+str(i)],2)
        R["b"+str(i)]=beta2*R["b"+str(i)]+(1-beta2)*np.power(lookahead_grads["db"+str(i)],2)

        R_c["W"+str(i)]=R["W"+str(i)]/(1-np.power(beta2,t))     # bias correction
        R_c["b"+str(i)]=R["b"+str(i)]/(1-np.power(beta2,t))
        params["W"+str(i)]-= (learning_rate*M_c["W"+str(i)])/(np.sqrt(R_c["W"+str(i)])+eps)
        params["b"+str(i)]-= (learning_rate*M_c["b"+str(i)])/(np.sqrt(R_c["b"+str(i)])+eps)
        params["W"+str(i)]-= learning_rate*(weight_decay/m)*params["W"+str(i)]
        params["b"+str(i)]-= learning_rate*(weight_decay/m)*params["b"+str(i)]
        
    return(params,M,R)

# Updates weights through Adam Gradient Descent
def UpdateWeightsAdam(m, params, gradients, layers, M, R, learning_rate, weight_decay, beta1, beta2, eps, t):
    M_c={}
    R_c={}
    for i in range(1,len(layers)):
        M["W"+str(i)]=beta1*M["W"+str(i)]+(1-beta1)*gradients["dw"+str(i)]
        M["b"+str(i)]=beta1*M["b"+str(i)]+(1-beta1)*gradients["db"+str(i)]

        M_c["W"+str(i)]=M["W"+str(i)]/(1-np.power(beta1,t))   # bias correction
        M_c["b"+str(i)]=M["b"+str(i)]/(1-np.power(beta1,t))
        R["W"+str(i)]=beta2*R["W"+str(i)]+(1-beta2)*np.power(gradients["dw"+str(i)],2)
        R["b"+str(i)]=beta2*R["b"+str(i)]+(1-beta2)*np.power(gradients["db"+str(i)],2)

        R_c["W"+str(i)]=R["W"+str(i)]/(1-np.power(beta2,t))    # bias correction
        R_c["b"+str(i)]=R["b"+str(i)]/(1-np.power(beta2,t))
        params["W"+str(i)]-= (learning_rate*M_c["W"+str(i)])/(np.sqrt(R_c["W"+str(i)])+eps)
        params["b"+str(i)]-= (learning_rate*M_c["b"+str(i)])/(np.sqrt(R_c["b"+str(i)])+eps)
        params["W"+str(i)]-= learning_rate*(weight_decay/m)*params["W"+str(i)]
        params["b"+str(i)]-= learning_rate*(weight_decay/m)*params["b"+str(i)]
    return(params,M,R)
