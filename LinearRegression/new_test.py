

FCF_0 = 7.398
r_WACC = 0.07401
g_FCF = 0.02
N = 12

def FCF(N):
    fcf_n = FCF_0 * (1+g_FCF)**N
    return fcf_n
 

def V_0(N):
    v0 = 0
    for n in range(1,N+1):
        term = FCF(n)/(1+r_WACC)**n
        v0 += term

    V_N = FCF(N+1)/(r_WACC-g_FCF)
    
    last_term = V_N/(1+r_WACC)**N
    return v0+last_term


for n in range(1, 12):
    v0 = V_0(N=n)
    print(f'V_0 : {v0}')