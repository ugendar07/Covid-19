import numpy as np
from matplotlib import pyplot as plt
import pandas as pd


def load_data(path, data_type):
    data = pd.read_csv(path)
    if data_type=="train":
        data = data.iloc[371:414,:]
    else:
        data= data.iloc[414:,:]
    data = data[['Confirmed','Tested','First Dose Administered','Recovered','Deceased']]
    data['Infections'] = data['Confirmed'] - data['Recovered'] - data['Deceased']
    return data.reset_index(drop=True)


def gradient_descent(beta, R_0, CIR_0, I_0, E_0  ,data, delta_c, N):
    epoch=100
    for _ in range(epoch):
        beta_g = calc_loss(beta + 0.01, R_0, CIR_0, I_0, E_0 ,data, delta_c, N)[0] - calc_loss(beta - 0.01 , R_0, CIR_0, I_0, E_0 ,data, delta_c, N)[0] 
        beta = beta - 0.00001*beta_g

        CIR_g = calc_loss(beta, R_0, CIR_0 + 0.1, I_0, E_0 ,data, delta_c, N)[0] - calc_loss(beta, R_0, CIR_0 - 0.1, I_0, E_0 ,data, delta_c, N)[0]
        CIR_0 = CIR_0 - 0.00001*CIR_g
        
        R_g = calc_loss(beta, R_0 + 1 , CIR_0, I_0, E_0 ,data, delta_c, N)[0] - calc_loss(beta, R_0 - 1  , CIR_0, I_0, E_0 ,data, delta_c, N)[0]
        R_0 = R_0 - (1)*R_g

        I_g = calc_loss(beta, R_0, CIR_0, I_0 + 1 , E_0 ,data, delta_c, N)[0] - calc_loss(beta, R_0, CIR_0,I_0 - 1  , E_0 ,data, delta_c, N)[0]
        I_0 = I_0 - (1)*I_g
    
        E_g = calc_loss(beta, R_0, CIR_0, I_0, E_0 + 1 ,data, delta_c, N)[0] - calc_loss(beta, R_0, CIR_0, I_0, E_0 - 1  ,data, delta_c, N)[0]
        E_0 = E_0 - (1)*E_g
    
    plotting([beta,R_0,CIR_0,I_0,E_0,data,delta_c,N],"No.oF Days","Log Scale Infections", 'Delta Curve',1)
    return [beta,R_0,CIR_0,I_0,E_0]          



def calc_theta(data,N):
    raw_data = pd.DataFrame(data, columns=['Infections']) 
    raw_data = raw_data.diff() 
    delta_cases = raw_data['Infections'].values[1:]
    temp = []
    for i in range(delta_cases.shape[0]):
        if(i<=6):
            temp.append(delta_cases[0:i+1].mean())
        else:
            temp.append(delta_cases[i-6:i+1].mean())
    temp = np.array(temp)
    return [1,32*N/100,14,0.35*N/100,0.5*N/100],temp



def calc_simulations(beta,R_0,CIR_0,I_0,E_0,data):   
    S_0 = N - I_0 - E_0 - R_0
    a,b = 1/5.8, 1/5    
    S, E, I, R = S_0, E_0, I_0, [R_0]
    for t in range(1,43):
        if(t<=29):
            W = R_0/30
        else:
            W = 0    
        S = S_0-beta*S_0*I_0/N+W
        E = E_0+beta*S_0*I_0/N-a*E_0
        I = I_0 +a*E_0-b*I_0
        R = np.append(R,np.array([R[t-1]+b*I_0-W]))
        S_0,E_0,I_0 = S, E, I
    return S,E,I,R,((data['Tested'].values[0]/data['Tested'].values)*CIR_0)[-1]             




def calc_loss(beta, R_0, CIR_0, I_0, E_0, data, delta_c, N):
    S_0 = N-I_0-E_0-R_0
    a,b = 1/5.8, 1/5
    S, E, I, R = S_0, E_0, [I_0], R_0
    for i in range(1,43): 
        if(i<=29):
            W = R_0/30 
        else:
            W = 0   
        I = np.append(I,np.array([I[i-1]+a*E_0-b*I[i-1]]))
        S = S_0-beta*S_0*I[i-1]/N+W
        E = E_0+beta*S_0*I[i-1]/N-a*E_0
        R = R_0+b*I[i-1]-W
        S_0,E_0,R_0 =S, E, R
    raw_data = pd.DataFrame(I/((data['Tested'].values[0]/data['Tested'].values)*CIR_0) , columns = ['Cases'])
    raw_data = raw_data.diff()

    delta_c1 = []
    temp = raw_data['Cases'].values[1:]
    for i in range(temp.shape[0]):
        if(i<=6):
            delta_c1.append(temp[0:i+1].mean())
        else:
            delta_c1.append(temp[i-6:i+1].mean())
            
    loss=np.sum((delta_c[:delta_c.shape[0]]-delta_c1[0:delta_c.shape[0]])**2)
    return loss**0.5, delta_c1

 

def open_loop_control(beta, S_0,E_0,I_0,R_0,CIR_0,eff,R_0_t,data , some = 0): 
    a,b = 1/5.8, 1/5
    S, E, I, R = S_0, E_0, [I_0], R_0
    raw_data = pd.DataFrame(data, columns=['First Dose Administered']) 
    vac = np.squeeze(raw_data.diff().fillna(200000).values)
    for t in range(1,146):
        if(t<=135):
            W = 0
        else:
            W = R_0_t[t-136] + eff*vac[t-136]    
        temp=S_0-beta*S_0*I[t-1]/N+ W-eff*vac[t-1]
        if(temp<0):
            S = 0
        else:
            S =temp
        E = E_0+beta*S_0*I[t-1]/N-a*E_0
        I = np.append(I,np.array([I[t-1]+a*E_0-b*I[t-1]]))
        R = R_0+b*I[t-1]-W+eff*vac[t-1]
        S_0,E_0,R_0 =S, E, R

    res = I/((data['Tested'].values[0]/data['Tested'].values)*CIR_0)
    if (some == 0):
        plotting([data,res],"No.oF Days","No.oF Infections", 'Predictions for Open Loop Control',0)
    elif (some == 2):
        plotting([data,res],"No.oF Days","No.oF Infections", 'Predictions for Open Loop Control 2*beta/3',0)

    elif (some == 3):
        plotting([data,res],"No.oF Days","No.oF Infections", 'Predictions for Open Loop Control beta/2',0)

    else:
        plotting([data,res],"No.oF Days","No.oF Infections", 'Predictions for Open Loop Control beta/3',0)



    return res                    


def closed_loop_control(beta, S_0, E_0, I_0, R_0, CIR_0, eff, R_0_t, data):
    CIR_t = (data['Tested'].values[0] / data['Tested'].values)
    CIR_t = CIR_t * CIR_0

    alpha, beta_inv = 1/5.8, 1/5
    S, E, I, R = S_0, E_0, [I_0], R_0

    raw_data = pd.DataFrame(data, columns=['First Dose Administered'])
    vaccinations = np.squeeze(raw_data.diff().fillna(200000).values)
    temp_1, temp_2 = beta, 0

    for t in range(1, 146):
        if t % 7 == 1:
            if t != 1 and temp_2 > 100001:
                temp_1 = beta / 3
            elif t != 1 and 25000 < temp_2 < 100000:
                temp_1 = beta / 2
            elif t != 1 and 10001 < temp_2 < 25000:
                temp_1 = beta * 0.66
            elif t != 1:
                temp_1 = beta
            temp_2 = 0

        if t <= 135:
            W = 0
        else:
            W = R_0_t[t - 136] + eff * vaccinations[t - 136]

        S = S_0 - temp_1 * S_0 * I[t - 1] / N + W - eff * vaccinations[t - 1]
        E = E_0 + temp_1 * S_0 * I[t - 1] / N - alpha * E_0
        I = np.append(I, np.array([I[t - 1] + alpha * E_0 - beta_inv * I[t - 1]]))
        R = R_0 + beta_inv * I[t - 1] - W + eff * vaccinations[t - 1]
        temp_2 += (alpha * E - beta_inv * I[t - 1]) / CIR_t[t - 1]

        S_0, E_0, R_0 = S, E, R

    res = I / CIR_t
    plotting([data, res], "No.oF Days", "No.oF Infections", 'Predictions for Closed Loop Control', 0)
    return res


def plotting(data, x, y, title, a):
    if a == 1:
        delta_c1 = calc_loss(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7])[1]
        plt.plot(np.linspace(0, 41, num=42), np.log(delta_c1), label="Model", linewidth=2, color='blue')
        plt.scatter(np.linspace(0, 41, num=42), np.log(data[6]), label="Data points", color='green')
    else:
        p = data[0]['Infections'].values.shape[0]
        plt.plot(np.linspace(0, p - 1, num=p), data[0]['Infections'].values, label='True Values', color='yellow')
        plt.plot(np.linspace(0, p - 1, num=p), data[1], label='Predicted Values', color='green')

    plt.xlabel(x)
    plt.ylabel(y)
    plt.legend()
    plt.title(title)
    plt.show()







if __name__ == '__main__':
    path = '../data/COVID19_data.csv'
    
    X_train = load_data(path,"train")
    X_test= load_data(path,"test")
    N=70000000

    theta=calc_theta(X_train,N)
    print("Training the model :")
    params = gradient_descent(theta[0][0],theta[0][1],theta[0][2],theta[0][3],theta[0][4],X_train,theta[1],N)  
    print("The Optimal Parameters:")
    print(params)
    print(params[1]/N)
    print(params[3]/N)
    print(params[4]/N)


    S, E, I, R_0_t, CIR_0 = calc_simulations(params[0],params[1],params[2],params[3],params[4],X_train)
    beta = params[0]
    eff = 0.66
    R =  R_0_t[-1]
    I = X_test['Infections'].values[0]*CIR_0


    print("Predictions of Open Loop Control:")

    a = open_loop_control(beta,S,E,I,R,CIR_0,eff,R_0_t,X_test)  
    b = open_loop_control(2*beta/3,S,E,I,R,CIR_0,eff,R_0_t,X_test, 2)  
    c = open_loop_control(beta/2,S,E,I,R,CIR_0,eff,R_0_t,X_test , 3)  
    d = open_loop_control(beta/3,S,E,I,R,CIR_0,eff,R_0_t,X_test , 4)  

    print("Predictions Of Closed Loop Control:")
    e = closed_loop_control(beta,S,E,I,R,CIR_0,eff,R_0_t,X_test)  
    

    plt.plot(a,label='beta open loop control')

    plt.plot(b,label=' 2 * beta/3 open loop control')

    plt.plot(c,label='beta/2 open loop control')

    plt.plot(d,label='beta/3 open loop control')

    plt.plot(e,label='beta closed loop control')

    plt.legend()
    plt.title("susceptible Population")
    plt.xlabel('No.oF days')
    plt.ylabel('No.oF suseptable people')
    plt.show()


