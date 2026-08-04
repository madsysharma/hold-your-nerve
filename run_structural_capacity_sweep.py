import logging, sys
import torch
import experiment
import cpn

l = logging.getLogger() 
l.addHandler(logging.StreamHandler(sys.stdout)) 
l.setLevel(logging.INFO)

LOG_DIR = "C:/Users/ejmil/OneDrive/Algoverse/agatha-ethm/models"  

# Remember: we should train EN on lesioned mrnn actuals, and train CPN on loss between EN predictions post stimulus and healthy brain data.
CAPACITY_LESION_CONFIGS = [
    ("m1", experiment.experiment.get_m1_lesion_config, True),
    ("aip", experiment.experiment.get_aip_lesion_config, True),
    ("f5_m1", experiment.experiment.get_aip_f5_con_lesion_config, False),
]

for lesion_name, lesion_config_fn, coadapt in CAPACITY_LESION_CONFIGS:
    for n in [20, 40, 87, 160]: #similar code to run_nocoadap
        cfg = lesion_config_fn(coadapt=coadapt, cuda="0", en_num_neurons=n)
        my_coproc = cpn.CPN_EN_CoProc(cfg, log_dir=f"{LOG_DIR}/capacity_{lesion_name}_{n}")
        my_experiment = experiment.experiment.stage(my_coproc, cfg)
        loss_history = my_experiment.run()

# Sweep for incorrect sigma: en_obs_sigma (not obs_sigma) mismatches the EN's view of M1 while CPN and true evaluation stay on the correct sigma.
for sigma in [0.5, 1.0, 1.75, 3.0, 5.0]:
    cfg = experiment.experiment.get_aip_f5_con_lesion_config(
        coadapt=False, cuda="0", en_obs_sigma=sigma
    )
    my_coproc = cpn.CPN_EN_CoProc(cfg, log_dir=f"{LOG_DIR}/sigma_{sigma}")
    my_experiment = experiment.experiment.stage(my_coproc, cfg)
    loss_history = my_experiment.run()
