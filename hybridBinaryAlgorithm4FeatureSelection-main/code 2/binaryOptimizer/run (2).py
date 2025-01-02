# -*- coding: utf-8 -*-
"""
Created on Wed May  4 11:42:52 2022

@author: 77132
"""
import statistics
from utils.FunctionUtil import *
from utils.Settings import *
from utils.GraphPlots import _plots_all_graphs
from utils.Parameters import ParameterRandomGrid
from utils.IOUtil import _save_results_to_csv__, _save_solutions_to_csv__, _save_search_process_data, _compute_solutions_metrics__
from physics_based.test_ArchOA import testArchOA
from physics_based.test_SA import testSA
from math_based.test_AOA import testAOA
from bio_based.test_EOSA import testEOSA
from bio_based.test_IEOSA import testIEOSA
from bio_based.test_SIRO import testSIRO
from bio_based.test_BBO import testBBO
from bio_based.test_IWO import testIWO
from bio_based.test_WHO import testWHO
from human_based.test_TLO import testTLO
from evolutionary_based.test_GA import testGA
from evolutionary_based.test_DE import testDE
from evolutionary_based.test_MA import testMA
from evolutionary_based.test_FPA import testFPA
from swarm_based.test_AO import testAO
from swarm_based.test_HGS import testHGS
from swarm_based.test_SSA import testSSA
from swarm_based.test_WOA import testWOA
from swarm_based.test_PSO import testPSO
from swarm_based.test_FFA import testFFA
from swarm_based.test_ABC import testABC
from swarm_based.test_GOA import testGOA
from swarm_based.test_SFO import testSFO
from swarm_based.test_MFO import testMFO
from utils.solution_scatter import SolutionScatter
import numpy as np
                   
algorithms=[
  #bio-based
  #'SIRO', 'EOSA', 'IEOSA', 
  'IWO', #'WHO', 'BBO',
  #evolution-based
  #'GA', 'DE', 'MA', 'FPA', 
  #math-based
  #'AOA', 
  #human-based
  #'TLO',
  #physics-based
  #'ArchOA', #'SA',
  #swarm-based
  #'AO', 'FFA', 'PSO', 'WOA', 'ABC', 'GOA', 'HGS', 'SFO', 'SSA', 'MFO',
  ]
ei=1
experiement=experiment_type[ei]
for algorithm in algorithms:
    params=ParameterRandomGrid()
    #benchmark_func=func_domain_ranges
    benchmark_func=ieee_func_domain_ranges
    save_results_dir=save_results_dir_CEC #save_results_dir,
    for func in benchmark_func:
        ranges, obj_func, f_sn, obj_func_name, min_val=benchmark_func[func]
        func_acronym=obj_func_name#f_sn+'_'+
        has_weight=False
        lb, ub=ranges
                        
        if (algorithm == 'IWO' and obj_func_name in ['C23', 'C24', 'C25', 'C26', 'C27', 'C28', 'C29', 'C30']):   
            
            all_best_fitness=[]
            all_measure=[]
            selected_best_position=None
            selectedmodel=None
            iterno=None        
            for run in range(number_of_runs):   
                if experiement=='generate_dataset':
                    for p in range(params.grid_count()):
                        #Get the current sir_param_grid
                        sir_model_rates_grid=params.get_params()
                        
                        #We need to formulate the label for each scatter image generated.
                        #This will allow to use this images as datasets in CNN model
                        grid_label='A'+str(p)+'_'+obj_func_name+'_Run'+str(run)
                        label='A'+str(p)
                        
                        #Print current experiement for user to see
                        print('Grid '+algorithm+': -> '+str(sir_model_rates_grid))
                        print('Experimenting '+algorithm+': -> run='+str(run)+' -> '+func)
                        
                        if algorithm == 'SIRO':
                            model, final_pop, solutions, best_position, best_fitness, ml_search_process=testSIRO(func, epoch[ei], problem_size, sir_model_rates_grid, lb, ub)
                        
                        #Since labels don't change in subsequent runs, we can simply just 
                        #store the labels only in the first run.
                        if run==0:
                            #Then store the label and its corrresponding sir_model_rates_grid values
                            filename='cnnlabels'
                            cnnlabelsavedir=save_results_dir+save_cnn_label_dir+algorithm+'/'
                            item={'label':label, 'sir_model_rates_grid':sir_model_rates_grid, 
                                  'all_fitness':solutions, 'best_fitness':best_fitness}
                            _save_results_to_csv__(item, filename, cnnlabelsavedir)
                        
                        #plot and save charts and scatter graphs
                        pathsavescatter=save_results_dir+save_scatter_dir+algorithm+'/'
                        scatter_params={"constraints": ranges, "r":0.25, "t":0.1, "obj_func": obj_func, 
                                        "solutions":final_pop,
                                "colors":[[0.0, 0.0, 0.0] for _ in range(problem_size)], 
                                "parameter_grid":grid_label, "obj_func_name":obj_func_name, 
                                "pathsave":pathsavescatter, "algorithm":algorithm
                            }
                        a_scatter = SolutionScatter(scatter_params)
                        a_scatter.save_plot() 
                
                    #Now that we are done with run[i], we must reset grid index so that next run[i+1]
                    #starts to select grids from params[0]
                    params.grid_reset()
                    
                elif experiement=='train_siro':
                    #sir_model_rates_grid=params.read_trained_params(params_save_results_dir)
                    sir_model_rates_grid={
                            "recruitment_rate": 0.3,
                            "disease_induced_death_rate": 0.07834286929910095,
                            "contact_rate_infectious": 0.3,
                            "contact_rate_recovered": 0.3,
                            "recovery_rate": 0.191426234123332,
                            "natural_death_rate": 0.12724382898779218,
                        }
                    if algorithm == 'SIRO':
                        model, final_pop, solutions, best_position, best_fitness, ml_search_process=testSIRO(func, epoch[ei], problem_size, sir_model_rates_grid, lb, ub)
                    if algorithm == 'EOSA':
                        model, final_pop, solutions, best_position, best_fitness, ml_search_process=testEOSA(func, epoch[ei], problem_size, modelrates, lb, ub, )                    
                    if algorithm == 'IEOSA':
                        model, final_pop, solutions, best_position, best_fitness, ml_search_process=testIEOSA(func, epoch[ei], problem_size, modelrates, lb, ub)
                    if algorithm == 'IWO':
                        model, final_pop, solutions, best_position, best_fitness, ml_search_process=testIWO(func, epoch[ei], problem_size, lb, ub)
                    if algorithm == 'WHO':
                        model, final_pop, solutions, best_position, best_fitness, ml_search_process=testWHO(func, epoch[ei], problem_size, lb, ub)
                    if algorithm == 'BBO':
                        model, final_pop, solutions, best_position, best_fitness, ml_search_process=testBBO(func, epoch[ei], problem_size, lb, ub)
                    
                    if algorithm == 'DE':
                        model, final_pop, solutions, best_position, best_fitness, ml_search_process=testDE(func, epoch[ei], problem_size, lb, ub)                        
                    if algorithm == 'GA':
                        model, final_pop, solutions, best_position, best_fitness, ml_search_process=testGA(func, epoch[ei], problem_size, lb, ub)                        
                    if algorithm == 'MA':
                        model, final_pop, solutions, best_position, best_fitness, ml_search_process=testMA(func, epoch[ei], problem_size, lb, ub)                        
                    if algorithm == 'FPA':
                        model, final_pop, solutions, best_position, best_fitness, ml_search_process=testFPA(func, epoch[ei], problem_size, lb, ub)                        
                    
                    if algorithm == 'TLO':
                        model, final_pop, solutions, best_position, best_fitness, ml_search_process=testTLO(func, epoch[ei], problem_size, lb, ub)                    
                        
                    if algorithm == 'AOA':
                        model, final_pop, solutions, best_position, best_fitness, ml_search_process=testAOA(func, epoch[ei], problem_size, lb, ub)                    
                    
                    if algorithm == 'ArchOA':
                        model, final_pop, solutions, best_position, best_fitness, ml_search_process=testArchOA(func, epoch[ei], problem_size, lb, ub)                    
                    if algorithm == 'SA':
                        model, final_pop, solutions, best_position, best_fitness, ml_search_process=testSA(func, epoch[ei], problem_size, lb, ub)                    
                                   
                    if algorithm == 'AO':
                        model, final_pop, solutions, best_position, best_fitness, ml_search_process=testAO(func, epoch[ei], problem_size, lb, ub)                    
                    if algorithm == 'HGS':
                        model, final_pop, solutions, best_position, best_fitness, ml_search_process=testHGS(func, epoch[ei], problem_size, lb, ub)
                        has_weight=True                        
                    if algorithm == 'SSA':
                        model, final_pop, solutions, best_position, best_fitness, ml_search_process=testSSA(func, epoch[ei], problem_size, lb, ub)                    
                    if algorithm == 'FFA':
                        model, final_pop, solutions, best_position, best_fitness, ml_search_process=testFFA(func, epoch[ei], problem_size, lb, ub)                    
                    if algorithm == 'PSO':
                        model, final_pop, solutions, best_position, best_fitness, ml_search_process=testPSO(func, epoch[ei], problem_size, lb, ub)
                        has_weight=True                        
                    if algorithm == 'WOA':
                        model, final_pop, solutions, best_position, best_fitness, ml_search_process=testWOA(func, epoch[ei], problem_size, lb, ub)                    
                    if algorithm == 'ABC':
                        model, final_pop, solutions, best_position, best_fitness, ml_search_process=testABC(func, epoch[ei], problem_size, lb, ub)                    
                    if algorithm == 'GOA':
                        model, final_pop, solutions, best_position, best_fitness, ml_search_process=testGOA(func, epoch[ei], problem_size, lb, ub)                    
                    if algorithm == 'SFO':
                        model, final_pop, solutions, best_position, best_fitness, ml_search_process=testSFO(func, epoch[ei], problem_size, lb, ub)                    
                    if algorithm == 'MFO':
                        model, final_pop, solutions, best_position, best_fitness, ml_search_process=testMFO(func, epoch[ei], problem_size, lb, ub)                    
                    
                    '''
                    No need to store ML data in this experiment
                    #save all search process for ML operations
                    MLdatasaveddir=save_results_dir+save_ML_data_dir+algorithm+'/'
                    _save_search_process_data(ml_search_process, func_acronym, MLdatasaveddir)
                    '''
                    
                    #Plot charts    
                    if run == 0:
                        chartsaveddir=save_results_dir+save_chart_dir+algorithm+'/'
                        selectedmodel=model
                        selected_best_position=best_position
                        iterno=run
                        _plots_all_graphs(model, chartsaveddir, run, func_acronym, has_weight, algorithm)
                    
                    #temporally store for use later in computing average
                    all_best_fitness.append(best_fitness)
            
                    #compute all
                    measure=_compute_solutions_metrics__(solutions, min_val)
                    all_measure.append(measure)
            
            #save all best solutions
            filename='best'
            bestsavedir=save_results_dir+save_bestsol_dir+algorithm+'/'
            item={'run':run, 
                  'func':func, 
                  'func_acronym':func_acronym, 
                  'best_solutions':selected_best_position, 
                  'fitness':statistics.mean(all_best_fitness)
                  }
            _save_results_to_csv__(item, filename, bestsavedir)
                        
            #save all solutions in history
            filename='agents'
            solnssavedir=save_results_dir+save_solutions_dir+algorithm+'/'
            _save_solutions_to_csv__(algorithm, func, func_acronym, has_weight, all_measure, model, filename, solnssavedir, number_of_runs)
                
'''
(algorithm == 'NoneAlgol' and 
             not(func in ['CEC_EbOA_7', 'CEC_5', 'CEC_EbOA_21', 'CEC_EbOA_20', 'CEC_2',
                          'CEC_EbOA_19', 'CEC_13', 'CEC_EbOA_18', 'CEC_3', 
                          'CEC_EbOA_17', 'whale_f11', 'hho_f12', 'hho_f13',
                          'CEC_EbOA_16', 'CEC_12', 'CEC_1', 'gCEC_EbOA_2','gCEC_1',
                          'CEC_EbOA_15','CEC_14','CEC_EbOA_14','CEC_EbOA_13',
                          'whale_f7','CEC_EbOA_12','CEC_EbOA_11','CEC_EbOA_10',
                          'CEC_EbOA_9','whale_f9','whale_f3','CEC_4','whale_f8',
                          'hho_f3', 'hho_f2','hho_f4','hho_f1','whale_f6',
                          'CEC_EbOA_8','CEC_EbOA_7','CEC_EbOA_5','C2',
                          'CEC_EbOA_6','CEC_EbOA_4','C4','C9','CEC_EbOA_3',
                          'CEC_EbOA_2','CEC_EbOA_1','CEC_6'])) or

'CEC_1', 'CEC_2', 'CEC_3', 'CEC_4', 'CEC_5',
'CEC_6', 'CEC_7', 'CEC_8', 'CEC_9', 'CEC_10',
'CEC_11', 'CEC_12', 'CEC_13',
'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10',
'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20',
'C21', 'C22', 'C23', 'C24', 'C25', 'C26', 'C27', 'C28', 'C29', 'C30',
'''