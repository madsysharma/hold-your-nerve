import logging, sys
import torch
import experiment
import cpn

l = logging.getLogger() #create a logger
l.addHandler(logging.StreamHandler(sys.stdout)) #logging.StreamHandler(sys.stdout) is a handler that writes logs to console
l.setLevel(logging.INFO)

LOG_DIR = "C:/Users/ejmil/OneDrive/Algoverse/agatha-ethm/models"  #see CPN_EN_CoProc in cpn.py

#see changes to get_stim_model()
#n = num neurons in EN
#experiment/config.py: declared as a field on Config, accepted as a parameter of get(), passed into the Config(...) constructor, 
#ends up stored as cfg.en_num_neurons
#cpn.py: read back out and used: num_neurons=cfg.en_num_neurons inside the call to stim_model.get_stim_model(...)
for n in [20, 40, 87, 160]: #similar code to run_nocoadap
    cfg = experiment.experiment.get_m1_lesion_config( #no coadapt for baseline
        coadapt=False, cuda="3", en_num_neurons=n
    )
    my_coproc = cpn.CPN_EN_CoProc(cfg, log_dir=f"{LOG_DIR}/capacity_{n}")
    my_experiment = experiment.experiment.stage(my_coproc, cfg)
    loss_history = my_experiment.run()

#sweep for incorrect sigma: en_obs_sigma (not obs_sigma) mismatches the
#EN's view of M1 while CPN and true evaluation stay on the correct sigma.
#Use get_aip_f5_con_lesion_config since it's the only lesion config that
#doesn't set drop_m1=True
#cant use if M1 is dropped
for sigma in [0.5, 1.0, 1.75, 3.0, 5.0]:
    cfg = experiment.experiment.get_aip_f5_con_lesion_config(
        coadapt=False, cuda="3", en_obs_sigma=sigma
    )
    my_coproc = cpn.CPN_EN_CoProc(cfg, log_dir=f"{LOG_DIR}/sigma_{sigma}")
    my_experiment = experiment.experiment.stage(my_coproc, cfg)
    loss_history = my_experiment.run()
