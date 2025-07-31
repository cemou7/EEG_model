from general_processor import Utils


def prodeceData_22ch_class4_filter():
    """
    Load PhysioNet motor imagery dataset, extract only 4 classes: 
    Left hand, Right hand, Feet, Tongue.
    """
    channels = ["Fz", "FC3", "FC1", "FCz", "FC2", "FC4", 
                "C5", "C3", "C1", "Cz", "C2", "C4", "C6", 
                "CP3", "CP1", "CPz", "CP2", "CP4", 
                "P1", "Pz", "P2", "POz"]
    
    data_path = "/home/work/CZT/CL-Model/dataset/PHYSIONET/original"
    exclude = [38, 88, 89, 92, 100, 104]
    subjects = [n for n in np.arange(1, 110) if n not in exclude]
    
    runs = [4, 6, 8, 10, 12, 14]  # contains all tasks
    save_path = "/home/work/CZT/CL-Model/dataset/Physionet_22ch"
    
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.mkdir(save_path)
    
    for sub in subjects:
        x, y = Utils.epoch(
            Utils.select_channels(
                Utils.filtering(
                    Utils.eeg_settings(
                        Utils.del_annotations(
                            Utils.concatenate_runs(
                                Utils.load_data(subjects=[sub], runs=runs, data_path=data_path)
                            )
                        )
                    )
                ), 
                channels, 
                allselect=False
            ),
            exclude_base=True,  # ✅ exclude baseline class
            num_class=4          # ✅ keep only 4 classes
        )
        np.save(os.path.join(save_path, f"x_sub_{sub}"), x, allow_pickle=True)
        np.save(os.path.join(save_path, f"y_sub_{sub}"), y, allow_pickle=True)
