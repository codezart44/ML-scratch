


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
        print(term)
        v0 += term

    V_N = FCF(N+1)/(r_WACC-g_FCF)
    print(V_N)
    print(v0)
    last_term = V_N/(1+r_WACC)**N
    v0 = v0 + last_term
    return v0

v0 = V_0(12)

print(f'V_0 : {v0}')

# print(186.026+67.086-134.392)