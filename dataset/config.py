def adjust_config(options=None):
    options.grid_size = 64
    if options.dataset == 'dfaust':
        options.input_dim = 3
        # options.grid_size = 48
        options.Ttot = 10
        options.Tcond = 3
        options.sample_rate = 5
        options.log_gif_num = 4
        # options.nepoch = 2000
        options.log_gif_every = 50
        options.lrate = 4e-4
        options.nkeypoints = 24
        options.local_const_weight = 0.001
        options.time_const_weight = 1.0
        options.graph_traj_weight = 1.0
        if options.grid_size == 48:
            options.firstdecay = 1200
            options.seconddecay = 2800
            options.nepoch = 4000
        elif options.grid_size == 64:
            options.firstdecay = 600
            options.seconddecay = 1400
            options.nepoch = 2000
        if options.pretrained_mode > 0:
            options.Ttot = 20
            options.Tcond = 5
            options.log_gif_num = 6
            options.nepoch = 2000
            options.log_gif_every = 200
            options.log_save_every = 50
    elif options.dataset == 'aist':
        options.is_eval = True
        options.input_dim = 3
        options.Ttot = 10
        options.Tcond = 3
        options.sample_rate = 2
        options.log_gif_num = 4
        # options.nepoch = 2000
        options.log_gif_every = 5
        options.lrate = 4e-4
        options.nkeypoints = 24
        options.local_const_weight = 0.001
        options.time_const_weight = 1.0
        options.graph_traj_weight = 1.0
        if options.grid_size == 48:
            options.firstdecay = 120
            options.seconddecay = 280
            options.nepoch = 400
        elif options.grid_size == 64:
            options.firstdecay = 60
            options.seconddecay = 140
            options.nepoch = 200
        if options.pretrained_mode > 0:
            options.Ttot = 20
            options.Tcond = 5
            options.log_gif_num = 6
            options.nepoch = 200
            options.log_gif_every = 20
    elif options.dataset == 'animals':
        options.input_dim = 3
        options.Ttot = 10
        options.Tcond = 3
        options.sample_rate = 1
        options.log_gif_num = 4
        # options.nepoch = 2000
        options.log_gif_every = 5
        options.lrate = 4e-4
        # options.nkeypoints = 48
        options.nkeypoints = 24
        options.gaussian_sigma = 2.0
        options.graph_traj_weight = 1e-6
        if options.grid_size == 48:
            options.firstdecay = 120
            options.seconddecay = 280
            options.nepoch = 400
        elif options.grid_size == 64:
            options.firstdecay = 120
            options.seconddecay = 170
            options.nepoch = 200
        if options.pretrained_mode > 0:
            options.Ttot = 20
            options.Tcond = 5
            options.log_gif_num = 6
            options.nepoch = 150
            options.log_gif_every = 5
    elif options.dataset == 'panda':
        options.is_eval = True
        options.input_dim = 3
        options.Ttot = 10
        options.Tcond = 3
        options.sample_rate = 1
        options.log_gif_num = 4
        # options.nepoch = 2000
        options.log_gif_every = 5
        options.lrate = 4e-4
        options.nkeypoints = 12
        options.local_const_weight = 1.0
        options.time_const_weight = 1.0
        options.graph_traj_weight = 0.001
        if options.grid_size == 48:
            options.firstdecay = 120
            options.seconddecay = 280
            options.nepoch = 400
        elif options.grid_size == 64:
            options.firstdecay = 60
            options.seconddecay = 140
            options.nepoch = 200
        if options.pretrained_mode > 0:
            options.log_gif_num = 6
            options.log_gif_every = 20
            options.Ttot = 20
            options.Tcond = 5
    elif options.dataset == 'hanco':
        options.is_eval = True
        options.input_dim = 3
        options.Ttot = 10
        options.Tcond = 3
        options.sample_rate = 1
        options.log_gif_num = 4
        # options.nepoch = 2000
        options.log_gif_every = 5
        options.lrate = 4e-4
        options.nkeypoints = 28
        options.gaussian_sigma = 1.0
        options.graph_traj_weight = 1e-6
        options.local_const_weight = 1.0
        options.vol_reg_weight = 0.1
        if options.grid_size == 48:
            options.firstdecay = 120
            options.seconddecay = 280
            options.nepoch = 400
        elif options.grid_size == 64:
            options.firstdecay = 120
            options.seconddecay = 170
            options.nepoch = 200
        if options.pretrained_mode > 0:
            options.Ttot = 20
            options.Tcond = 5
            options.log_gif_num = 6
            options.nepoch = 200
            options.log_gif_every = 20
    else:
        raise ValueError("Wrong Dataset Assignment!")

    if options.pretrained_mode > 0:
        options.firstdecay = 1e10
        options.seconddecay = 1e10


    return options
