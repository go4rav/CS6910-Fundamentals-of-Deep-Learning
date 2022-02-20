


def updateWeightsMomentum(params,gradients,layers,alpha,beta,M):
    for i in range(1,len(layers)):
        M["W"+str(i)]=beta*V["W"+str(i)]+(1-beta)*gradients["dw"+str(i)]
        M["b"+str(i)]=beta*V["b"+str(i)]+(1-beta)*gradients["db"+str(i)]
        params["W"+str(i)]-= alpha* M["W"+str(i)]
        params["b"+str(i)]-= alpha*M["b"+str(i)]
    return(params)

def updateWeightsAdam(params,gradients,layers,alpha,gamma1,gamma2,eps,t,M,R):
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
        params["W"+str(i)]-= (alpha*M_c["W"+str(i)])/(np.sqrt(R_c["W"+str(i)])+eps)
        params["b"+str(i)]-= (alpha*M_c["b"+str(i)])/(np.sqrt(R_c["b"+str(i)])+eps)
    return(params)


def updateWeightsRMS(params,gradients,layers,alpha,beta,R):
    for i in range(1,len(layers)):
        R["W"+str(i)]=beta*R["W"+str(i)]+(1-beta)*np.power(gradients["dw"+str(i)],2)
        R["b"+str(i)]=beta*R["b"+str(i)]+(1-beta)*np.power(gradients["db"+str(i)],2)
        params["W"+str(i)]-= alpha* R["W"+str(i)]
        params["b"+str(i)]-= alpha* R["b"+str(i)]
    return(params)


def updateWeightsNAdam(params, lookahead_grads, layers, alpha,gamma1,gamma2,eps,t):
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
        params["W"+str(i)]-= (alpha*M_c["W"+str(i)])/(np.sqrt(R_c["W"+str(i)])+eps)
        params["b"+str(i)]-= (alpha*M_c["b"+str(i)])/(np.sqrt(R_c["b"+str(i)])+eps)
        
        
        
  def updateWeights(params, gradients, layers, alpha,gamma1,gamma2,eps,t):
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
        params["W"+str(i)]-= (alpha*M_c["W"+str(i)])/(np.sqrt(R_c["W"+str(i)])+eps)
        params["b"+str(i)]-= (alpha*M_c["b"+str(i)])/(np.sqrt(R_c["b"+str(i)])+eps)
        
        
  def updateWeightsNAG(params,lookahead_grads,layers,alpha,beta):
    
    for i in range(1,len(layers)):
        M["W"+str(i)]=beta*M["W"+str(i)]+(1-beta)*lookahead_grads["dw"+str(i)]
        M["b"+str(i)]=beta*M["b"+str(i)]+(1-beta)*lookahead_grads["db"+str(i)]
        params["W"+str(i)]-= alpha* M["W"+str(i)]
        params["b"+str(i)]-= alpha*M["b"+str(i)] 
        
        
 def updateWeightsRMS(params, gradients, layers, alpha, beta ,eps):
    for i in range(1,len(layers)):
        R["W"+str(i)]=beta*R["W"+str(i)]+(1-beta)*np.power(gradients["dw"+str(i)],2)
        R["b"+str(i)]=beta*R["b"+str(i)]+(1-beta)*np.power(gradients["db"+str(i)],2)
        params["W"+str(i)]-= (alpha*gradients["dw"+str(i)])/(np.sqrt(R["W"+str(i)])+eps)
        params["b"+str(i)]-= (alpha*gradients["db"+str(i)])/(np.sqrt(R["b"+str(i)])+eps)
