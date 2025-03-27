import numpy as np
import sys
from typing import Tuple, Optional , Union, Dict ,Literal
import torch
import pandas as pd
import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import time
from time import time


# Wrapper function to compute t-DCF for a fixed ASV system and a unconstrained ASV system.
def compute_t_dcf(bonafide_score_cm : np.ndarray, spoof_score_cm : np.ndarray, Prior_spoof : float,
                  target_scores : np.ndarray,nontarget_scores : np.ndarray,spoof_scores : np.ndarray,list_asv_score : list,type : str):
    
    if type != 'constrained' and type != 'unconstrained' and type != 'constrained_ver2':
        raise ValueError('type must be either "constrained" or "unconstrained" or "constrained_ver2"')
    
    # Parameters of the t-DCF cost model
    if type == 'constrained':
        cost_model = {
            'Ptar':(1-Prior_spoof)*0.99,         # Prior probability of target speaker
            'Pnon':(1-Prior_spoof)*0.01,         # Prior probability of nontarget speaker (zero-effort impostor)
            'Pspoof': Prior_spoof,        # Prior probability of spoofing attack
            'Cmiss_asv': 1,      # Cost of ASV falsely rejecting target
            'Cfa_asv': 10,       # Cost of ASV falsely accepting nontarget
            'Cmiss_cm': 1,       # Cost of CM falsely rejecting target
            'Cfa_cm': 10          # Cost of CM falsely accepting spoof
        }
    elif type == 'unconstrained':
        cost_model = {
            'Ptar':(1-Prior_spoof)*0.99,         # Prior probability of target speaker
            'Pnon':(1-Prior_spoof)*0.01,         # Prior probability of nontarget speaker (zero-effort impostor)
            'Pspoof': Prior_spoof,              # Prior probability of spoofing attack
            'Cmiss': 1,
            'Cfa': 10,                          # Cost of ASV falsely accepting nontarget
            'Cfa_spoof':10,
        }
    elif type == 'constrained_ver2':
        cost_model = {
            'Ptar':(1-Prior_spoof)*0.99,         # Prior probability of target speaker
            'Pnon':(1-Prior_spoof)*0.01,         # Prior probability of nontarget speaker (zero-effort impostor)
            'Pspoof': Prior_spoof,        # Prior probability of spoofing attack
            'Cmiss': 1,  
            'Cfa': 10,       # Cost of ASV falsely accepting nontarget
            'Cfa_spoof':10,
        }
    list_tDCF_norm = []
    list_CM_thresholds = []
    list_tdcf = []
    for idx,asv_score in enumerate(list_asv_score):

        Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, Pfa_spoof_asv = obtain_asv_error_rates(target_scores,nontarget_scores,spoof_scores,asv_score)

        # Flag to print a summary of the cost parameters and the t-DCF cost function
        print_cost = True
        print("The t-DCF evaluation type is:",type)
        if type == 'constrained':
            tDCF_norm, CM_thresholds,tdcf = compute_tDCF_constrained(bonafide_score_cm, spoof_score_cm, Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, cost_model, print_cost)
        elif type == 'unconstrained':
            tDCF_norm, CM_thresholds,tdcf = compute_tDCF_Unconstrained(bonafide_score_cm, spoof_score_cm, Pfa_asv, Pmiss_asv, Pmiss_spoof_asv,Pfa_spoof_asv, cost_model, print_cost)
        elif type == 'constrained_ver2':
            tDCF_norm, CM_thresholds,tdcf = compute_tDCF_constrained_ver2(bonafide_score_cm, spoof_score_cm, Pfa_asv, Pmiss_asv, Pmiss_spoof_asv,Pfa_spoof_asv, cost_model, print_cost)
        
        list_tdcf.append(tdcf)
        list_tDCF_norm.append(tDCF_norm)
        list_CM_thresholds.append(CM_thresholds)

        if idx == 0:
            print("the asv threshold is from eer on asv development set ",asv_score)
        
        if idx == 1:
            print("the asv threshold is from eer on asv evaluation set ",asv_score)
        elif idx > 1:
            print("the asv threshold is from eer on asv set ",asv_score)
        
        print("the CM thresholds is: ",CM_thresholds)

        print("the CM threshold min is:",CM_thresholds[np.argmin(tDCF_norm)])

        print("the tDCF_norm is:",tDCF_norm)

        print("the min tDCF_norm is:",min(tDCF_norm))
        
        print("the tDCF is:",tdcf)
        
        print("the min tdcf is:",min(tdcf))
        
    return list_tDCF_norm,list_CM_thresholds,list_tdcf

def obtain_asv_error_rates(tar_asv : np.ndarray , non_asv : np.ndarray, spoof_asv : np.ndarray, asv_threshold : float):
    
    # False alarm and miss rates for ASV
    Pfa_asv = np.sum(non_asv >= asv_threshold) / non_asv.size
    Pmiss_asv = np.sum(tar_asv < asv_threshold) / tar_asv.size

    # Rate of rejecting spoofs in ASV
    if spoof_asv.size == 0:
        Pmiss_spoof_asv = None
        Pfa_spoof_asv = None
    else:
        Pmiss_spoof_asv = np.sum(spoof_asv < asv_threshold) / spoof_asv.size
        
        # False Acceptance Rate for Spoofing Attacks
        Pfa_spoof_asv = np.sum(spoof_asv >= asv_threshold) / spoof_asv.size

    return Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, Pfa_spoof_asv


def compute_det_curve(target_scores : np.ndarray, nontarget_scores : np.ndarray):

    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate((np.ones(target_scores.size), np.zeros(nontarget_scores.size)))

    # Sort labels based on scores
    indices = np.argsort(all_scores, kind='mergesort')
    labels = labels[indices]

    # Compute false rejection and false acceptance rates
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - (np.arange(1, n_scores + 1) - tar_trial_sums)

    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / target_scores.size))  # false rejection rates
    far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size))  # false acceptance rates
    thresholds = np.concatenate((np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))  # Thresholds are the sorted scores

    return frr, far, thresholds

# Implementation of the EER function 
def compute_eer(target_scores : np.ndarray, nontarget_scores : np.ndarray):
    """ Returns equal error rate (EER), the corresponding threshold and the false rejection and false acceptance rates. """
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    return eer, thresholds[min_index],frr, far


def compute_tDCF_constrained(bonafide_score_cm : np.ndarray, spoof_score_cm : np.ndarray, Pfa_asv : float, Pmiss_asv : float, Pmiss_spoof_asv: float, cost_model : Dict[str, float], print_cost : bool):
    """
    Compute Tandem Detection Cost Function (t-DCF) [1] for a fixed ASV system.
    In brief, t-DCF returns a detection cost of a cascaded system of this form,

      Speech waveform -> [CM] -> [ASV] -> decision

    where CM stands for countermeasure and ASV for automatic speaker
    verification. The CM is therefore used as a 'gate' to decided whether or
    not the input speech sample should be passed onwards to the ASV system.
    Generally, both CM and ASV can do detection errors. Not all those errors
    are necessarily equally cost, and not all types of users are necessarily
    equally likely. The tandem t-DCF gives a principled with to compare
    different spoofing countermeasures under a detection cost function
    framework that takes that information into account.

    INPUTS:

      bonafide_score_cm   A vector of POSITIVE CLASS (bona fide or human)
                          detection scores obtained by executing a spoofing
                          countermeasure (CM) on some positive evaluation trials.
                          trial represents a bona fide case.
      spoof_score_cm      A vector of NEGATIVE CLASS (spoofing attack)
                          detection scores obtained by executing a spoofing
                          CM on some negative evaluation trials.
      Pfa_asv             False alarm (false acceptance) rate of the ASV
                          system that is evaluated in tandem with the CM.
                          Assumed to be in fractions, not percentages.
      Pmiss_asv           Miss (false rejection) rate of the ASV system that
                          is evaluated in tandem with the spoofing CM.
                          Assumed to be in fractions, not percentages.
      Pmiss_spoof_asv     Miss rate of spoof samples of the ASV system that
                          is evaluated in tandem with the spoofing CM. That
                          is, the fraction of spoof samples that were
                          rejected by the ASV system.
      cost_model          A struct that contains the parameters of t-DCF,
                          with the following fields.

                          Ptar        Prior probability of target speaker.
                          Pnon        Prior probability of nontarget speaker (zero-effort impostor)
                          Psoof       Prior probability of spoofing attack.
                          Cmiss_asv   Cost of ASV falsely rejecting target.
                          Cfa_asv     Cost of ASV falsely accepting nontarget.
                          Cmiss_cm    Cost of CM falsely rejecting target.
                          Cfa_cm      Cost of CM falsely accepting spoof.

      print_cost          Print a summary of the cost parameters and the
                          implied t-DCF cost function?

    OUTPUTS:

      tDCF_norm           Normalized t-DCF curve across the different CM
                          system operating points; see [2] for more details.
                          Normalized t-DCF > 1 indicates a useless
                          countermeasure (as the tandem system would do
                          better without it). min(tDCF_norm) will be the
                          minimum t-DCF used in ASVspoof 2019 [2].
      CM_thresholds       Vector of same size as tDCF_norm corresponding to
                          the CM threshold (operating point).

    NOTE:
    o     In relative terms, higher detection scores values are assumed to
          indicate stronger support for the bona fide hypothesis.
    o     You should provide real-valued soft scores, NOT hard decisions. The
          recommendation is that the scores are log-likelihood ratios (LLRs)
          from a bonafide-vs-spoof hypothesis based on some statistical model.
          This, however, is NOT required. The scores can have arbitrary range
          and scaling.
    o     Pfa_asv, Pmiss_asv, Pmiss_spoof_asv are in fractions, not percentages.

    References:

      [1] T. Kinnunen, K.-A. Lee, H. Delgado, N. Evans, M. Todisco,
          M. Sahidullah, J. Yamagishi, D.A. Reynolds: "t-DCF: a Detection
          Cost Function for the Tandem Assessment of Spoofing Countermeasures
          and Automatic Speaker Verification", Proc. Odyssey 2018: the
          Speaker and Language Recognition Workshop, pp. 312--319, Les Sables d'Olonne,
          France, June 2018 (https://www.isca-speech.org/archive/Odyssey_2018/pdfs/68.pdf)

      [2] ASVspoof 2019 challenge evaluation plan
          TODO: <add link>
    """


    # Sanity check of cost parameters
    if cost_model['Cfa_asv'] < 0 or cost_model['Cmiss_asv'] < 0 or \
            cost_model['Cfa_cm'] < 0 or cost_model['Cmiss_cm'] < 0:
        print('WARNING: Usually the cost values should be positive!')

    if cost_model['Ptar'] < 0 or cost_model['Pnon'] < 0 or cost_model['Pspoof'] < 0 or \
            np.abs(cost_model['Ptar'] + cost_model['Pnon'] + cost_model['Pspoof'] - 1) > 1e-10:
        sys.exit('ERROR: Your prior probabilities should be positive and sum up to one.')

    # Unless we evaluate worst-case model, we need to have some spoof tests against asv
    if Pmiss_spoof_asv is None:
        sys.exit('ERROR: you should provide miss rate of spoof tests against your ASV system.')

    # Sanity check of scores
    combined_scores = np.concatenate((bonafide_score_cm, spoof_score_cm))
    if np.isnan(combined_scores).any() or np.isinf(combined_scores).any():
        sys.exit('ERROR: Your scores contain nan or inf.')

    # Sanity check that inputs are scores and not decisions
    n_uniq = np.unique(combined_scores).size
    if n_uniq < 3:
        sys.exit('ERROR: You should provide soft CM scores - not binary decisions')

    # Obtain miss and false alarm rates of CM
    Pmiss_cm, Pfa_cm, CM_thresholds = compute_det_curve(bonafide_score_cm, spoof_score_cm)

    # Constants - see ASVspoof 2019 evaluation plan
    C1 = cost_model['Ptar'] * (cost_model['Cmiss_cm'] - cost_model['Cmiss_asv'] * Pmiss_asv) - \
         cost_model['Pnon'] * cost_model['Cfa_asv'] * Pfa_asv
    C2 = cost_model['Cfa_cm'] * cost_model['Pspoof'] * (1 - Pmiss_spoof_asv)

    # Sanity check of the weights
    if C1 < 0 or C2 < 0:
        sys.exit('You should never see this error but I cannot evalute tDCF with negative weights - please check whether your ASV error rates are correctly computed?')

    # Obtain t-DCF curve for all thresholds
    tDCF = C1 * Pmiss_cm + C2 * Pfa_cm

    # Normalized t-DCF
    tDCF_norm = tDCF / np.minimum(C1, C2)

    # Everything should be fine if reaching here.
    if print_cost:

        print('t-DCF evaluation from [Nbona={}, Nspoof={}] trials\n'.format(bonafide_score_cm.size, spoof_score_cm.size))
        print('t-DCF MODEL')
        print('   Ptar         = {:8.5f} (Prior probability of target user)'.format(cost_model['Ptar']))
        print('   Pnon         = {:8.5f} (Prior probability of nontarget user)'.format(cost_model['Pnon']))
        print('   Pspoof       = {:8.5f} (Prior probability of spoofing attack)'.format(cost_model['Pspoof']))
        print('   Cfa_asv      = {:8.5f} (Cost of ASV falsely accepting a nontarget)'.format(cost_model['Cfa_asv']))
        print('   Cmiss_asv    = {:8.5f} (Cost of ASV falsely rejecting target speaker)'.format(cost_model['Cmiss_asv']))
        print('   Cfa_cm       = {:8.5f} (Cost of CM falsely passing a spoof to ASV system)'.format(cost_model['Cfa_cm']))
        print('   Cmiss_cm     = {:8.5f} (Cost of CM falsely blocking target utterance which never reaches ASV)'.format(cost_model['Cmiss_cm']))
        print('\n   Implied normalized t-DCF function (depends on t-DCF parameters and ASV errors), s=CM threshold)')

        if C2 == np.minimum(C1, C2):
            print('   tDCF_norm(s) = {:8.5f} x Pmiss_cm(s) + Pfa_cm(s)\n'.format(C1 / C2))
        else:
            print('   tDCF_norm(s) = Pmiss_cm(s) + {:8.5f} x Pfa_cm(s)\n'.format(C2 / C1))

    return tDCF_norm, CM_thresholds,tDCF



def compute_tDCF_Unconstrained(bonafide_score_cm : np.ndarray, spoof_score_cm : np.ndarray, Pfa_asv : float, Pmiss_asv : float, Pmiss_spoof_asv : float,Pfa_spoof_asv : float, cost_model : Dict[str, float] , print_cost : bool):
    """
    This function computes the t-DCF cost function for a unconstrained ASV system.
    The function is based on the following paper:
    "Tandem Assessment of Spoofing Countermeasures and Automatic Speaker Verification: Fundamentals", Tomi Kinnunen, Hector Delgado, Nicholas Evans, et al., 
    IEEE/ACM Transactions on Audio, Speech, and Language Processing, 2020.
    """
   
    # Sanity check of cost parameters
    if cost_model['Cfa'] < 0 or cost_model['Cmiss'] < 0 or \
            cost_model['Cfa'] < 0 or cost_model['Cmiss'] < 0:
        print('WARNING: Usually the cost values should be positive!')

    if cost_model['Ptar'] < 0 or cost_model['Pnon'] < 0 or cost_model['Pspoof'] < 0 or \
            np.abs(cost_model['Ptar'] + cost_model['Pnon'] + cost_model['Pspoof'] - 1) > 1e-10:
        sys.exit('ERROR: Your prior probabilities should be positive and sum up to one.')

    # Unless we evaluate worst-case model, we need to have some spoof tests against asv
    if Pmiss_spoof_asv is None:
        sys.exit('ERROR: you should provide miss rate of spoof tests against your ASV system.')

    # Sanity check of scores
    combined_scores = np.concatenate((bonafide_score_cm, spoof_score_cm))
    if np.isnan(combined_scores).any() or np.isinf(combined_scores).any():
        sys.exit('ERROR: Your scores contain nan or inf.')

    # Sanity check that inputs are scores and not decisions
    n_uniq = np.unique(combined_scores).size
    if n_uniq < 3:
        sys.exit('ERROR: You should provide soft CM scores - not binary decisions')

    # Obtain miss and false alarm rates of CM
    Pmiss_cm, Pfa_cm, CM_thresholds = compute_det_curve(bonafide_score_cm, spoof_score_cm)

    P_a = (1-Pmiss_cm) * Pmiss_asv
    P_b = (1-Pmiss_cm) * Pfa_asv
    P_c = Pfa_cm*Pfa_spoof_asv
    P_d = Pmiss_cm
    
    # Obtain t-DCF curve for all thresholds
    tDCF = cost_model['Cmiss']*cost_model['Ptar']*(P_a +P_d) + cost_model['Cfa']*cost_model['Pnon']*P_b + cost_model['Cfa_spoof']*cost_model['Pspoof']*P_c

    # Normalized t-DCF
    min1 = cost_model['Cfa']*cost_model['Pnon'] + cost_model['Cfa_spoof']*cost_model['Pspoof']
    min2 = cost_model['Cmiss']*cost_model['Ptar']
    
    tDCF_norm = tDCF / np.minimum(min1, min2)
    
      # Everything should be fine if reaching here.
    if print_cost:

        print('t-DCF evaluation from [Nbona={}, Nspoof={}] trials\n'.format(bonafide_score_cm.size, spoof_score_cm.size))
        print('t-DCF MODEL')
        print('   Ptar         = {:8.5f} (Prior probability of target user)'.format(cost_model['Ptar']))
        print('   Pnon         = {:8.5f} (Prior probability of nontarget user)'.format(cost_model['Pnon']))
        print('   Pspoof       = {:8.5f} (Prior probability of spoofing attack)'.format(cost_model['Pspoof']))
        print('   Cfa          = {:8.5f} (Cost of ASV falsely accepting a nontarget)'.format(cost_model['Cfa']))
        print('   Cmiss        = {:8.5f} (Cost of ASV falsely rejecting target speaker)'.format(cost_model['Cmiss']))
        print('   Cfa_spoof    = {:8.5f} (Cost of CM falsely passing a spoof to ASV system)'.format(cost_model['Cfa_spoof']))
        print('\n   Implied normalized t-DCF function (depends on t-DCF parameters and ASV errors), s=CM threshold)')

     
    return tDCF_norm, CM_thresholds,tDCF



def compute_tDCF_constrained_ver2(bonafide_score_cm : np.ndarray, spoof_score_cm : np.ndarray, Pfa_asv  : float, Pmiss_asv  : float, Pmiss_spoof_asv  : float,Pfa_spoof_asv  : float, cost_model : Dict[str, float],print_cost : bool):
    """
    This function computes the t-DCF cost function for a fixed ASV system (include C_0 constant). The function is based on the following paper:
    "Tandem Assessment of Spoofing Countermeasures and Automatic Speaker Verification: Fundamentals", Tomi Kinnunen, Hector Delgado, Nicholas Evans, et al., 
    IEEE/ACM Transactions on Audio, Speech, and Language Processing, 2020.
    """
   
    # Sanity check of cost parameters
    if cost_model['Cfa'] < 0 or cost_model['Cmiss'] < 0 or \
            cost_model['Cfa'] < 0 or cost_model['Cmiss'] < 0:
        print('WARNING: Usually the cost values should be positive!')

    if cost_model['Ptar'] < 0 or cost_model['Pnon'] < 0 or cost_model['Pspoof'] < 0 or \
            np.abs(cost_model['Ptar'] + cost_model['Pnon'] + cost_model['Pspoof'] - 1) > 1e-10:
        sys.exit('ERROR: Your prior probabilities should be positive and sum up to one.')

    # Unless we evaluate worst-case model, we need to have some spoof tests against asv
    if Pmiss_spoof_asv is None:
        sys.exit('ERROR: you should provide miss rate of spoof tests against your ASV system.')

    # Sanity check of scores
    combined_scores = np.concatenate((bonafide_score_cm, spoof_score_cm))
    if np.isnan(combined_scores).any() or np.isinf(combined_scores).any():
        sys.exit('ERROR: Your scores contain nan or inf.')

    # Sanity check that inputs are scores and not decisions
    n_uniq = np.unique(combined_scores).size
    if n_uniq < 3:
        sys.exit('ERROR: You should provide soft CM scores - not binary decisions')

    # Obtain miss and false alarm rates of CM
    Pmiss_cm, Pfa_cm, CM_thresholds = compute_det_curve(bonafide_score_cm, spoof_score_cm)

    C0 = cost_model['Ptar']*cost_model['Cmiss']*Pmiss_asv + cost_model['Pnon']*cost_model['Cfa']*Pfa_asv
    C1 = cost_model['Ptar'] * (cost_model['Cmiss'] - cost_model['Cmiss'] * Pmiss_asv) - \
         cost_model['Pnon'] * cost_model['Cfa'] * Pfa_asv
    C2 = cost_model['Cfa_spoof'] * cost_model['Pspoof'] * Pfa_spoof_asv
    # Obtain t-DCF curve for all thresholds
    tDCF = C0 +C1*Pmiss_cm +C2*Pfa_cm

    # Normalized t-DCF

    tDCF_norm = tDCF / (C0 + np.minimum(C1, C2))
    
      # Everything should be fine if reaching here.
    if print_cost:

        print('t-DCF evaluation from [Nbona={}, Nspoof={}] trials\n'.format(bonafide_score_cm.size, spoof_score_cm.size))
        print('t-DCF MODEL')
        print('   Ptar         = {:8.5f} (Prior probability of target user)'.format(cost_model['Ptar']))
        print('   Pnon         = {:8.5f} (Prior probability of nontarget user)'.format(cost_model['Pnon']))
        print('   Pspoof       = {:8.5f} (Prior probability of spoofing attack)'.format(cost_model['Pspoof']))
        print('   Cfa          = {:8.5f} (Cost of ASV falsely accepting a nontarget)'.format(cost_model['Cfa']))
        print('   Cmiss        = {:8.5f} (Cost of ASV falsely rejecting target speaker)'.format(cost_model['Cmiss']))
        print('   Cfa_spoof    = {:8.5f} (Cost of CM falsely passing a spoof to ASV system)'.format(cost_model['Cfa_spoof']))
        print('\n   Implied normalized t-DCF function (depends on t-DCF parameters and ASV errors), s=CM threshold)')

     
    return tDCF_norm, CM_thresholds,tDCF


# DCF function for ASV evaluation
def dcf_formula(Pfa_asv : float,Pmiss_asv : float,Prior_target : float,Prior_non_target : float,cost_model_dcf : Dict[str, float] , is_print: bool):
    dcf = cost_model_dcf['Cmiss_asv']*Prior_target*Pmiss_asv + cost_model_dcf['Cfa_asv']*Prior_non_target*Pfa_asv
    if is_print:
        print("The DCF is: ",dcf)
    return dcf

# Implementation of the EER function 
def compute_eer_3(y : np.ndarray, y_score: np.ndarray): # y is the ground truth, y_score is the prediction score
    """ 
    function to compute the equal error rate with interpolation1d and brentq.
    """
    fpr, tpr, thresholds = roc_curve(y, y_score, pos_label=1) # pos_label=1 means that the positive class is 1  

    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr,kind='linear')(x), 0., 1.) # brentq is a root finding algorithm
    thresh = interp1d(fpr, thresholds)(eer) # interp1d is a linear interpolation function
    return eer, thresh # eer is the equal error rate, thresh is the threshold at eer

# Implementation of the EER function 
def compute_eer_2(label : np.ndarray, pred_score : np.ndarray, positive_label : int = 1):
    # all fpr, tpr, fnr, fnr, threshold are lists (in the format of np.array)
    fpr, tpr, threshold = roc_curve(label, pred_score, pos_label=positive_label)
    fnr = 1 - tpr

    # the threshold of fnr == fpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]

    # theoretically eer from fpr and eer from fnr should be identical but they can be slightly differ in reality
    eer_1 = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    eer_2 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]

    #print(eer_1)
    #print(eer_2)
    # return the mean of eer from fpr and from fnr
    eer = (eer_1 + eer_2) / 2
    return eer_1,eer_2,eer,eer_threshold


# function to plot DET curve with EER point
def DETCurve(fpr: float,fnr: float, eer_fpr: float,eer_fnr: float, model_name: str, plot_type: Literal["plot", "save"] = "plot", is_mlflow: bool = False):
    plt.figure()
    FPR= fpr 
    TPR =1-fnr
    FNR = fnr
    if plot_type == "plot":
        plt.plot(FPR*100, FNR*100, color='b', lw=2, label='DET Curve')
    elif plot_type == "step":
        plt.step(FPR*100, FNR*100, color='b', lw=2, label='DET Curve')
        

    # Plot EER point
    #eer_index = np.argmin(np.abs(FPR - (1 - TPR)))
    #eer_fpr = FPR[eer_index]*100
    #eer_fnr = FNR[eer_index]*100
    eer_fpr = eer_fpr*100
    eer_fnr = eer_fnr*100
    
    if plot_type == "plot":
        plt.plot(eer_fpr, eer_fnr, 'ro', label=f'EER ({eer_fpr:.4f}%, {eer_fnr:.4f}%)')
    elif plot_type == "step":
        plt.step(eer_fpr, eer_fnr, 'ro', label=f'EER ({eer_fpr:.4f}%, {eer_fnr:.4f}%)')

    # Annotate EER point
    plt.annotate(f'EER ({eer_fpr:.4f}%, {eer_fnr:.4f}%)',
                xy=(eer_fpr, eer_fnr),
                xytext=(eer_fpr - eer_fpr/4, eer_fnr + eer_fnr/4),
                arrowprops=dict(arrowstyle='->'))
    plt.xlabel('False Positive Rate (FPR) (%)')
    plt.ylabel('False Negative Rate (FNR) (%)')
    plt.title('Detection Error Tradeoff (DET) Curve (%) with model name - ' + model_name)
    plt.legend()
    plt.grid(True)


if __name__ == "__main__":
    """
    This script simulates the evaluation of speaker verification and spoofing countermeasure systems
    by computing evaluation metrics such as EER, DCF, and t-DCF using synthetic data.
    """

    np.random.seed(42)

    # Simulate ASV scores
    n_target = 300
    n_nontarget = 300
    n_spoof = 300

    target_scores = np.random.normal(2, 1, n_target)
    nontarget_scores = np.random.normal(0, 1, n_nontarget)
    spoof_scores = np.random.normal(-1, 1, n_spoof)

    # Simulate CM scores
    n_bonafide = 300
    n_spoof_cm = 300
    bonafide_score_cm = np.random.normal(1, 1, n_bonafide)
    spoof_score_cm = np.random.normal(-1, 1, n_spoof_cm)

    # Compute EERs:
    eer, eer_threshold, frr, far = compute_eer(target_scores, nontarget_scores)
    print(f"\n=== ASV EER ===\nEER: {eer:.4f}, Threshold: {eer_threshold:.4f}")

    eer1, eer2, eer_avg, eer_thresh2 = compute_eer_2(
        np.concatenate([np.ones(n_target), np.zeros(n_nontarget)]),
        np.concatenate([target_scores, nontarget_scores])
    )
    print(f"\n=== ASV EER v2 ===\nEER1: {eer1:.4f}, EER2: {eer2:.4f}, Avg EER: {eer_avg:.4f}, Threshold: {eer_thresh2:.4f}")

    eer3, thresh3 = compute_eer_3(
        np.concatenate([np.ones(n_target), np.zeros(n_nontarget)]),
        np.concatenate([target_scores, nontarget_scores])
    )
    print(f"\n=== ASV EER v3 ===\nEER: {eer3:.4f}, Threshold: {thresh3:.4f}")

    # Define ASV thresholds to test (EER threshold + a few variants)
    list_asv_score = [eer_threshold, eer_thresh2, eer_thresh2 + 0.5]

    # Prior spoofing probability
    Prior_spoof = 0.05

    print("\n=== t-DCF Evaluation ===")
    list_tDCF_norm, list_CM_thresholds, list_tdcf = compute_t_dcf(
        bonafide_score_cm,
        spoof_score_cm,
        Prior_spoof,
        target_scores,
        nontarget_scores,
        spoof_scores,
        list_asv_score,
        type="constrained"  # Can also test "unconstrained" or "constrained_ver2"
    )

    # Evaluate the DCF at the first ASV threshold as an example
    print("\n=== DCF Evaluation ===")
    asv_threshold_example = list_asv_score[0]
    Pfa_asv, Pmiss_asv, _, _ = obtain_asv_error_rates(
        target_scores, nontarget_scores, spoof_scores, asv_threshold_example
    )
    Prior_target = 0.95
    Prior_non_target = 0.05
    cost_model_dcf = {'Cmiss_asv': 1, 'Cfa_asv': 10}

    dcf = dcf_formula(Pfa_asv, Pmiss_asv, Prior_target, Prior_non_target, cost_model_dcf, is_print=True)
    print(f"DCF: {dcf:.4f}")

    # Optional: Plot t-DCF curves
    for i, (tdcf_norm, cm_thres) in enumerate(zip(list_tDCF_norm, list_CM_thresholds)):
        plt.figure()
        plt.plot(cm_thres, tdcf_norm, label=f't-DCF Norm {i}')
        plt.xlabel('CM threshold')
        plt.ylabel('Normalized t-DCF')
        plt.title(f't-DCF Curve for ASV Threshold {i}')
        plt.legend()
        plt.grid(True)
        plt.show()
