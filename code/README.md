# How to run these code

To run the whole project, we first have to collect data. To collect data for training, run the following in this directory
    
    bash scripts/run_collect_push.sh
for pushing task

    bash scripts/run_collect_pull.sh
for pulling task. Add --test in the script files when collecting test data, and you might use xvfb to run this code if you have no screen.

After finishing the data collection, run

    bash scripts/run_train_AAP.sh
to initialize the training of AAP module. Mind that you may have to change the name of datalist in the script file.
Then run

    bash scripts/run_train_AIP.sh
to initialize the training of AIP module.

After initializing the training of both AAP and AIP module, run

    bash scripts/run_train_iter.sh
to iteratively train these two module.

Finally, run

    bash scripts/run_train_AAP_aff.sh
    bash scripts/run_train_AIP_aff.sh
to train the affordance maps in both modules.



