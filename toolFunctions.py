import numpy as np
import scipy.special as sp
import scipy.signal as sg
import scipy.integrate as integrate

class UtilFunc():
    def smooth(self, tn, rs, rr, smthLength=9, smthPol=2, flag=True):
        if flag:
            if smthPol>=smthLength:
                smthPol=smthLength-1
            tn = sg.savgol_filter(tn, smthLength, smthPol)
            rs = sg.savgol_filter(rs, smthLength, smthPol)
            rr = sg.savgol_filter(rr, smthLength, smthPol)
        return tn, rs, rr

    def coarsen_array(self, a, level=2, method='mean'):
        # coarsen code from spinmob
        """
        Returns a coarsened (binned) version of the data. Currently supports
        any of the numpy array operations, e.g. min, max, mean, std, ...

        level=2 means every two data points will be binned.
        level=0 or 1 just returns a copy of the array
        """
        if a is None: return None

        # make sure it's a numpy array
        a = np.array(a)

        # quickest option
        if level in [0, 1, False]: return a

        # otherwise assemble the python code to execute
        code = 'a.reshape(-1, level).' + method + '(axis=1)'

        # execute, making sure the array can be reshaped!
        try:
            return eval(code, dict(a=a[0:int(len(a) / level) * level], level=level))
        except:
            print("ERROR: Could not coarsen array with method " + repr(method))
            return a

class EchoFunctions():
    @staticmethod
    def pllFit(t, xoff, t2, gPara, fm, phi, n, a, c):
        return a * (np.exp(-np.power(t / (t2 / 1e6), n)) * np.cos(2 * np.pi * gPara * 1e6 * (
            np.sin((t - xoff) * 2 * np.pi * fm * 1e3 + phi / 180 * np.pi) / (2 * np.pi * fm * 1e3) - 2 * np.sin(
                1 / 2 * (t - xoff) * 2 * np.pi * fm * 1e3 + phi / 180 * np.pi) / (2 * np.pi * fm * 1e3) + np.sin(
                phi / 180 * np.pi) / (2 * np.pi * fm * 1e3)))) + c

    @staticmethod
    def besselFit(t, xoff, t2, gPara, fm, n, a, c):
        return a * np.exp(-np.power(t / (t2 / 1e6), n)) * \
               (sp.jv(0, 8 * np.pi * (gPara * 1e6 / (2 * np.pi * fm * 1e3)) * np.power(
                   np.sin(2 * np.pi * fm * 1e3 * (t - xoff) / 4), 2))) + c


# input: gPara in angular MHz, fm in khz, rbz=0
    @staticmethod
    def besselHalfFit(t, t2, gPara, fm, phi, n, a, c):
        iResult = 0
        num = 100
        totalPhi = 180 # in degree
        phis = np.linspace(phi, phi+totalPhi, num)
        dphi = totalPhi/num

        for iphi in phis:
            iResult = iResult + dphi*EchoFunctions.probPhiHalf(t, gPara=gPara, fm=fm, phi=iphi, a=a, c=c, n=n, t2=t2)
        return iResult/(totalPhi)

    @staticmethod
    def probPhiHalf(t, gPara, fm, phi, t2, a, c, n):
        return a*(np.exp(-np.power(t / (t2 / 1e6), n))
                    *np.cos(EchoFunctions.theta(t, gPara=1e6*gPara, gPerp=0, wm=2*np.pi*1e3*fm,
                                                phi=np.pi/180*phi, rbz=0)))+c

    @staticmethod
    def prob(t, gPara=0*1e6, gPerp=0*1e6, fm=500, phi=0.0, t2=10.0, rbz=10.0, a=0.1, c=0.9, n=1):
        # 여기서 parameter의 유닛을 맞춘다.
        # 아래 return에서 theta함수를 호출할 때 단위 변경된 값을 입력한다.
        # phi[deg->rad], fm->wm, gPara[MHz], gPerp[MHz ->angular Hz], rBz[MHz->angular Hz]
        # 시간 t는 이미 second 단위로 입력되고 있다.
        return a*(np.exp(-np.power(t / (t2 / 1e6), n))
                    *np.cos(EchoFunctions.theta(t, gPara=1e6*gPara, gPerp=2*np.pi*1e6*gPerp, wm=2*np.pi*1e3*fm,
                                                phi=np.pi/180*phi, rbz=2*np.pi*1e6*rbz)))+c

    @staticmethod
    def theta(t, gPara=0.0, gPerp=0.0, wm=5e7, phi=0.0, rbz=10.0):
        return EchoFunctions.thetaPara(t, gPara=gPara, wm=wm, phi=phi) #\
                #+EchoFunctions.thetaPerp(t, gPerp=gPerp, wm=wm, phi=phi, rbz=rbz)


    @staticmethod
    def thetaPara(t, gPara=0.0, wm=5e7, phi=0.0):
        # Gpara는 ncomms5429 논문대로 non-angular frequency를 함수 내에서 사용한다.
        return 2*np.pi*gPara*(np.sin(t*wm + phi)/wm - 2*np.sin(1/2*t*wm + phi)/wm + np.sin(phi)/wm)
        #return 4*gPara/wm*(np.sin(wm*t/8))**2*np.sin(wm*t/2+phi)

    @staticmethod
    def thetaPerp(t, gPerp=0.0, wm=5e7, phi=0.0, rbz=10.0):
        if gPerp == 0 and rbz==0:
            R=1
            isPerp = 0
        else:
            R = np.sqrt((rbz) ** 2 + gPerp ** 2)
            isPerp = 1
        return isPerp*R/wm*(
            2*sp.ellipeinc(1/2*wm*t+phi, (gPerp/R)**2)-sp.ellipeinc(wm*t+phi, (gPerp/R)**2)
            - sp.ellipeinc(phi, (gPerp/R)**2)

        )
