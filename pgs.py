from obspy.taup import TauPyModel
import numpy as np
from pyproj import Geod
import datetime, warnings, pickle, utm, time, cartopy, toml, dask, pdb
from obspy import UTCDateTime
from obspy.signal.trigger import recursive_sta_lta
from obspy.taup import taup_create
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from matplotlib.dates import date2num, num2date
from matplotlib import gridspec
from scipy.integrate import simpson as simps    # Changed from simps to simpson
from scipy.integrate import trapezoid as trapz  # Changed from trapz to trapezoid
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io import shapereader
from multiprocessing import Pool
from dask.distributed import Client, progress
warnings.filterwarnings("ignore")

def make_model_file(config, del_dist=5, p_phase=['P','p'], s_phase=['S','s'],
                    threads_per_worker=1, n_workers=12):
    '''
    Makes a model file for a given configuration, containing predicted travel times
    and backazimuths for each spatial grid point

    del_dist defines the spacing in kilometers between each point on the travel time curve
    '''

    # Defining source grid and set of distances:
    x, y, z = make_source_grid(config['grid']['bounds'],
                               config['grid']['N_lon'],
                               config['grid']['N_lat'],
                               config['grid']['N_d'])
    grid_dist_max = get_max_distance_gridnode(config['stations']['stlo'],
                                              config['stations']['stla'],
                                              config['grid']['bounds'])
    grid_dists = np.arange(0, grid_dist_max+del_dist, del_dist)
    
    # Computing P and S wave traveltimes for each distance:
    try:
        model = TauPyModel(model=config['earth_model'].split('.')[0])
    except:
        print('Earth model may not exist and needs to be generated from tvel file')
        pdb.set_trace()
    grid_times_p, grid_times_s = make_ttcurves(model, grid_dists, z,
                                               p_phases=p_phase,
                                               s_phases=s_phase,
                                               threads_per_worker=threads_per_worker,
                                               n_workers=n_workers)

    # Computing traveltimes and backazimuths for each source grid point:
    t_p, t_s = compute_traveltimes_grid(config['grid']['N_lat'], config['grid']['N_lon'],
                                        config['grid']['N_d'], x, y,
                                        config['stations']['stlo'],
                                        config['stations']['stla'],
                                        grid_dists, del_dist, grid_times_p, grid_times_s,
                                        threads_per_worker=threads_per_worker,
                                        n_workers=n_workers)
    b = compute_backazimuths_grid(config['grid']['N_lat'], config['grid']['N_lon'], x, y,
                                  config['stations']['stlo'], config['stations']['stla'])
    
    # Saving predictions for source grid:
    data = [t_p, t_s, b]
    pickle.dump(data, open(config['model_file'], 'wb'))

def make_source_grid(bounds, N_lon, N_lat, N_d):
    '''
    Generates a source grid
    
    Inputs:
    bounds is a list with: [min_lon, max_lon, min_lat, max_lat, min_dep, max_dep]
    N_lon, N_lat, and N_d are the number of discrete points in lon, lat and depth
    
    Outputs:
    x, y are matrices containing horizontal nodes, with dimension [N_lon, N_lat]
    z is a vector of vertical nodes, with dimension [N_d]
    '''
    
    # Computing source grid:
    lons = np.linspace(bounds[0], bounds[1], N_lon)
    lats = np.linspace(bounds[2], bounds[3], N_lat)
    x, y = np.meshgrid(lons, lats)
    z = np.linspace(bounds[4], bounds[5], N_d)

    return x, y, z

def get_max_distance_gridnode(stlo, stla, bounds):
    '''
    Returns the maximum distance to a grid node (km) given a set of stations
    and bounds
    '''

    g = Geod(ellps='sphere')

    grid_dists = []
    for i in range(0, len(stla)):
        _,_,d1 = g.inv(bounds[0],bounds[2],stlo[i],stla[i])
        _,_,d2 = g.inv(bounds[0],bounds[3],stlo[i],stla[i])
        _,_,d3 = g.inv(bounds[1],bounds[2],stlo[i],stla[i])
        _,_,d4 = g.inv(bounds[1],bounds[3],stlo[i],stla[i])
        dmax = np.max((d1,d2,d3,d4))/1000
        grid_dists.append(dmax)
    grid_dist_max = np.ceil(np.max(grid_dists))
    
    return grid_dist_max


def _compute_tt_for_dist(arg, z, model, grid_dists, p_phase_list, s_phase_list):
    '''
    (PRIVATE) Computes predicted P and S traveltimes for a given distance defined by
    grid_dists[arg] and a range of depths defined by z
    '''
    
    grid_times_p = np.zeros(len(z))
    grid_times_s = np.zeros(len(z))
    for j in range(0, len(z)):
        arrivals_p = model.get_travel_times(source_depth_in_km=z[j],
                                    distance_in_degree=grid_dists[arg]/111.1,
                                    phase_list=p_phase_list)
        grid_times_p[j] = arrivals_p[0].time
        arrivals_s = model.get_travel_times(source_depth_in_km=z[j],
                                    distance_in_degree=grid_dists[arg]/111.1,
                                    phase_list=s_phase_list)
        grid_times_s[j] = arrivals_s[0].time
    
    return grid_times_p, grid_times_s

def make_ttcurves(model, grid_dists, z, p_phases=["P","p"], s_phases=["S","s"],
                  threads_per_worker=1, n_workers=12):
    '''
    Computes predicted travel time curves for a range of distances and depths

    Inputs:
    model is a TauPyModel
    grid_dists is a vector of distances (km)
    z is a vector of depths (km)

    Outputs
    p-wave travel times for each distance and depth [N_distances, N_depths]
    s-wave travel times for each distance and depth [N_distances, N_depths]
    '''

    p_phase_list = p_phases; s_phase_list = s_phases

    # Create and manage client with context manager
    with Client(threads_per_worker=threads_per_worker, n_workers=n_workers) as client:
        # Building delayed loop
        lazy_results = []
        for i in range(0, len(grid_dists)):
            lazy_result = dask.delayed(_compute_tt_for_dist)(i, z, model, grid_dists, p_phase_list, s_phase_list)
            lazy_results.append(lazy_result)
        
        # Running loop with dask
        res = dask.compute(*lazy_results)

    # Process results outside the context manager
    res = np.array(res)
    grid_times_p = res[:,0,:]
    grid_times_s = res[:,1,:]

    return grid_times_p, grid_times_s

def _predict_times_for_station(arg, N_lat, N_lon, N_d, x, y, stlo, stla, grid_dists, del_dist, grid_times_p, grid_times_s):
    '''
    (PRIVATE) Computes predicted arrival time at a single station (arg provides the index) for each grid node by
    interpolating between arrival times computed as a function of distance
    '''

    g = Geod(ellps='sphere')

    t_p = np.zeros((N_lat, N_lon, N_d))
    t_s = np.zeros((N_lat, N_lon, N_d))
    for i in range(0, N_lat):
        for j in range(0, N_lon):
            for k in range(0, N_d):
                az12,az21,dist = g.inv(x[i,j],y[i,j],stlo[arg],stla[arg])
                dist = dist/1000
                ix = np.argmin(np.abs(dist - grid_dists))    # Finding the index of grid_dists that is closest to dist
                
                # Interpolating between pre-computed values for grid point:
                dist_resid = dist - grid_dists[ix]
                if dist_resid > 0:
                    del_t_p = grid_times_p[ix+1,k] - grid_times_p[ix,k]
                    grad_p = del_t_p/del_dist
                    t_p[i,j,k] = grid_times_p[ix,k] + (dist_resid * grad_p)
                    del_t_s = grid_times_s[ix+1,k] - grid_times_s[ix,k]
                    grad_s = del_t_s/del_dist
                    t_s[i,j,k] = grid_times_s[ix,k] + (dist_resid * grad_s)
                elif dist_resid < 0:
                    del_t_p = grid_times_p[ix,k] - grid_times_p[ix-1,k]
                    grad_p = del_t_p/del_dist
                    t_p[i,j,k] = grid_times_p[ix-1,k] + (np.abs(dist_resid) * grad_p)
                    del_t_s = grid_times_s[ix,k] - grid_times_s[ix-1,k]
                    grad_s = del_t_s/del_dist
                    t_s[i,j,k] = grid_times_s[ix-1,k] + (np.abs(dist_resid) * grad_s)
                else:
                    t_p[i,j,k] = grid_times_p[ix,k]
                    t_s[i,j,k] = grid_times_s[ix,k]
    
    return t_p, t_s

def compute_traveltimes_grid(N_lat, N_lon, N_d, x, y, stlo, stla, grid_dists, del_dist, grid_times_p, grid_times_s,
                             threads_per_worker=1, n_workers=12):
    '''
    Computes traveltimes for each grid node and station location by interpolating 1D travel time curves
    '''
    
    N_s = len(stla)

    # Create and manage client with context manager
    with Client(threads_per_worker=threads_per_worker, n_workers=n_workers) as client:
        lazy_results = []
        for i in range(0, N_s):
            lazy_result = dask.delayed(_predict_times_for_station)(i, N_lat, N_lon, 
                                   N_d, x, y, stlo, stla, grid_dists, del_dist, grid_times_p, grid_times_s)
            lazy_results.append(lazy_result)
        
        # Running loop with dask
        res = dask.compute(*lazy_results)

    # Process results outside the context manager
    res = np.array(res)
    t_p = res[:,0,:,:,:]
    t_s = res[:,1,:,:,:]

    return t_p, t_s

def compute_backazimuths_grid(N_lat, N_lon, x, y, stlo, stla):
    '''
    Computes backazimuths for each spatial grid node and station location
    '''

    g = Geod(ellps='sphere')

    N_s = len(stla)

    b = np.zeros((N_s, N_lat, N_lon))
    for l in range(0, N_s):
        for i in range(0, N_lat):
            for j in range(0, N_lon):
                az12,az21,dist = g.inv(x[i,j],y[i,j],stlo[l],stla[l])
                b[l,i,j] = az21 % 360
    
    return b

def compute_predictions(config, evla, evlo, evdp, evt0, std_p=0, std_s=0, std_b=0):
    '''
    Computes synthetic observed arrival times and backazimuths for a given model, event, indices
    of stations that observed the event (stid_event), stations, and specified uncertainties, where:
    - std_p is the standard deviation of Gaussian errors to add to P-wave arrivals
    - std_s is the standard deviation of Gaussian errors to add to S-wave arrivals
    - std_b is the standard deviation of Gaussian errors to add to backazimuth measurements
    '''
    
    g = Geod(ellps='sphere')
    
    try:
        model = TauPyModel(model=config['earth_model'].split('.')[0])
    except:
        print('Earth model has not been created!')
        pdb.set_trace()
    
    stla = config['stations']['stla']
    stlo = config['stations']['stlo']
    N_sta = len(stla)
    
    # Initialize output vectors:
    t_a_p = np.zeros(N_sta)
    t_a_s = np.zeros(N_sta)
    b_a = np.zeros(N_sta)
    
    # Creating perturbations to add to predictions:
    t_noi_p = np.random.normal(0, std_p, N_sta)
    t_noi_s = np.random.normal(0, std_s, N_sta)
    b_noi = np.random.normal(0, std_b, N_sta)
    
    for i in range(0, N_sta):
        az12,az21,dist = g.inv(evlo,evla,stlo[i],stla[i])
        dist = dist/1000
        arrivals_p = model.get_travel_times(source_depth_in_km=evdp,
                                      distance_in_degree=dist/111.1,
                                      phase_list=["P","p"])
        arrivals_s = model.get_travel_times(source_depth_in_km=evdp,
                                      distance_in_degree=dist/111.1,
                                      phase_list=["S","s"])
        t_a_p[i] = evt0 + (arrivals_p[0].time + t_noi_p[i])/86400
        t_a_s[i] = evt0 + (arrivals_s[0].time + t_noi_s[i])/86400
        b_a[i] = (az21 + b_noi[i]) % 360.
    
    # Converting config to a string and adding predictions:
    config = toml.dumps(config)
    config = config + '\n'
    config = config + '[data]'
    config = config + '\n' + 't_p = ' + str([str(UTCDateTime(num2date(t_a_p_i))) for t_a_p_i in t_a_p])
    config = config + '\n' + 't_s = ' + str([str(UTCDateTime(num2date(t_a_s_i))) for t_a_s_i in t_a_s])
    config = config + '\n' + 'b = ' + str([str(b_a_i) for b_a_i in b_a])

    # Converting config to TOML and writing to file:
    config = toml.loads(config)
    with open(config['model_file'].replace('.model','.toml'), 'w') as f:
        toml.dump(config, f)
    

def gaussian(residual, std):
    '''
    Returns the value of a Gaussian distribution given a residual and
    standard deviation, std
    '''
    
    gauss = (1/np.sqrt(2*np.pi*std**2)) * np.exp(-0.5 * (residual/std)**2)    # Adding normalization term
    #gauss = np.exp(-0.5 * (residual/std)**2)                                 # For Gaussian errors
    #gauss = np.exp(-np.abs(residual)/std)                                    # For Laplacian errors
    
    return gauss

def _get_likelihood_for_event_hypothesis(i, j, k, t0, t_p, t_s, t_a_p, t_a_s, b, b_a, use_p, use_s, use_b, t_std, b_std):
    '''
    Returns the likelihood for an event hypothesis defined by (i, j, k, t0)
    
    i = index of longitude (integer)
    j = index of latitude (integer)
    k = index of depth (integer)
    t0 = origin time (float in datenum format)
    
    tt = travel time predictions (NumPy array of dimensions [N_s, N_lat, N_lon, N_d])
    
    t_a = arrival times
    '''
    
    # Loop over stations:
    likl = 1
    for l in range(0, len(t_a_p)):

        # Compute predicted arrival times:
        tpp = t0 + t_p[l, i, j, k]/86400
        tps = t0 + t_s[l, i, j, k]/86400
        
        # ---------
        # Define whether to use P, S, or backazimuth for the specific station:
        use_p_l = use_p; use_s_l = use_s; use_b_l = use_b
        # Turning off P, S, or backazimuth if the measurement is None for specific station:
        if not(t_a_p[l]):
            use_p_l = False
        if not(t_a_s[l]):
            use_s_l = False
        if not(b_a[l]):
            use_b_l = False
        # ---------
        if use_p_l:
            res_p = (t_a_p[l] - tpp)*86400
        if use_s_l:
            res_s = (t_a_s[l] - tps)*86400
        if use_b_l:
            angle_diff = (b_a[l] - b[l, i, j])
            res_b = (angle_diff+180) % 360 - 180

        if use_p_l:
            if use_s_l:
                if use_b_l:
                    likl = likl * gaussian(res_p, t_std) * gaussian(res_s, t_std) * gaussian(res_b, b_std)
                else:
                    likl = likl * gaussian(res_p, t_std) * gaussian(res_s, t_std)
            else:
                if use_b_l:
                    likl = likl * gaussian(res_p, t_std) * gaussian(res_b, b_std)
                else:
                    likl = likl * gaussian(res_p, t_std)
        else:
            if use_s_l:
                if use_b_l:
                    likl = likl * gaussian(res_s, t_std) * gaussian(res_b, b_std)
                else:
                    likl = likl * gaussian(res_s, t_std)
            else:
                if use_b_l:
                    likl = likl * gaussian(res_b, b_std)
    
    return likl

def _get_likelihood_for_origintime(arg, N_lat, N_lon, N_d, t0s, t_p, t_s, t_a_p, t_a_s, t_std, use_p, use_s, b, b_a, use_b, b_std):
    '''
    Calls get_likelihood_for_event_hypothesis multiple times for each
    event hypothesis for a specific origin time
    '''
    
    t0 = t0s[arg]
    
    likl = np.zeros((N_lat, N_lon, N_d))
    
    for i in range(0, N_lat):
        for j in range(0, N_lon):
            for k in range(0, N_d):
                likl[i,j,k] = _get_likelihood_for_event_hypothesis(i, j, k, t0, t_p, t_s, t_a_p, t_a_s, b, b_a, use_p, use_s, use_b, t_std, b_std)
    
    return likl

def do_location(config, t_std=0.2, b_std=5, use_p = True, use_s=True, use_b=False,
                threads_per_worker=1, n_workers=12):
    '''
    Performs location over a range of origin times t0s_in using travel time predictions (and spatial
    grid nodes) defined in prediction_file, using station indices defined by stid_event and observed
    (or synthetic) data

    Optional parameters include the allowed standard deviation in arrival time, s_std_in, and
    Boolean flags that indicate whether to use P waves and/or S waves

    Output:
    likl is a 4D matrix of likelihoods with dimensions [N_origin times, N_lat, N_lon, N_d]
    '''

    # Reading data from prediction_file:
    data = pickle.load(open(config['model_file'], 'rb'))
    t_p = data[0]; t_s = data[1]; b = data[2]

    # Reading parameters from configuration:
    bounds = config['grid']['bounds']; N_lon = config['grid']['N_lon']
    N_lat = config['grid']['N_lat']; N_d = config['grid']['N_d']
    t_start = date2num(UTCDateTime(config['time-grid']['t_start']).datetime)
    t_end = date2num(UTCDateTime(config['time-grid']['t_end']).datetime)
    t0s = np.linspace(t_start,t_end,config['time-grid']['N_t'])

    t_a_p = []
    for t_p_i in config['data']['t_p']:
        if t_p_i == '':
            t_a_p.append(None)
        else:
            t_a_p.append(date2num(UTCDateTime(t_p_i).datetime))

    t_a_s = []
    for t_s_i in config['data']['t_s']:
        if t_s_i == '':
            t_a_s.append(None)
        else:
            t_a_s.append(date2num(UTCDateTime(t_s_i).datetime))

    b_a = []
    for b_i in config['data']['b']:
        if b_i == '':
            b_a.append(None)
        else:
            b_a.append(float(b_i))
    
    # Create and manage client with context manager
    with Client(threads_per_worker=threads_per_worker, n_workers=n_workers) as client:
        lazy_results = []
        for i in range(0, len(t0s)):
            lazy_result = dask.delayed(_get_likelihood_for_origintime)(
                i, N_lat, N_lon, N_d, t0s, t_p, t_s, t_a_p, t_a_s, 
                t_std, use_p, use_s, b, b_a, use_b, b_std)
            lazy_results.append(lazy_result)
        
        # Running loop with dask
        res = dask.compute(*lazy_results)
        likl = np.array(res)

    return likl

def fix_event_to_gridnode(evla, evlo, evdp, evt0, t0s, prediction_file):
    '''
    Returns the nearest grid node solution for a synthetic event

    (Useful for testing synthetic simulations)
    '''

    # Reading data from prediction_file:
    data = pickle.load(open(prediction_file, 'rb'))
    t_p = data[0]; t_s = data[1]; b = data[2]; bounds = data[3]; N_lon = data[4]; N_lat = data[5]; N_d = data[6]
    lons = np.linspace(bounds[0], bounds[1], N_lon)
    lats = np.linspace(bounds[2], bounds[3], N_lat)
    z = np.linspace(bounds[4], bounds[5], N_d)

    evla = lats[np.argmin(np.abs(lats - evla))]
    evlo = lons[np.argmin(np.abs(lons - evlo))]
    evdp = z[np.argmin(np.abs(z - evdp))]
    evt0 = t0s[np.argmin(np.abs(t0s - evt0))]

    return evla, evlo, evdp, evt0

def print_maximum_likelihood_solution(likl, t0s, prediction_file):
    '''
    Prints the maximum likelihood solution
    '''

    # Reading data from prediction_file:
    data = pickle.load(open(prediction_file, 'rb'))
    t_p = data[0]; t_s = data[1]; b = data[2]; bounds = data[3]; N_lon = data[4]; N_lat = data[5]; N_d = data[6]
    lons = np.linspace(bounds[0], bounds[1], N_lon)
    lats = np.linspace(bounds[2], bounds[3], N_lat)
    z = np.linspace(bounds[4], bounds[5], N_d)

    k_, i_, j_, l_ = np.unravel_index(likl.argmax(), likl.shape)

    print('Optimal event hypothesis:')
    print('Time =', num2date(t0s[k_]))
    print('longitude =', lats[i_])
    print('latitude =', lons[j_])
    print('Depth =', z[l_])

def contour_threshold(marg_p, lons, lats, cred=0.95, n_trials=1000):
    '''
    Computes the threshold of the marginal posterior distribution, marg_p, given a specified
    credibility, cred, and a number of trials (affects the precision)
    '''

    marg_p_trial_thresholds = np.linspace(0, np.max(marg_p)/2, n_trials)
    vols = np.zeros(len(marg_p_trial_thresholds))
    for i in range(0, len(marg_p_trial_thresholds)):
        marg_p_i = marg_p.copy()
        marg_p_i[marg_p_i < marg_p_trial_thresholds[i]] = 0.
        vols[i] = simps(simps(marg_p_i, lats), lons)
    i = np.argmin(np.abs(vols - cred))
    
    return marg_p_trial_thresholds[i]

def plot_marginal_distributions(config, likl, aspect1=None, aspect2=None,
                             cmap='hot_r', scale_bar_length=5, scale_bar_position=(0.85,0.05),
                             override_extent=None, plot_maxlikl=False, plot_baz=True,
                             plot_true=False, evla=None, evlo=None, evdp=None, 
                             plot_credibility=True, cred=0.95, do_smoothing=False, smoothing_width=1,
                             shp_file=None):
    '''
    Plots the full 3D marginal distribution over location and depth
    '''

    
    g = Geod(ellps='sphere')

    # Reading data from prediction_file:
    data = pickle.load(open(config['model_file'], 'rb'))
    t_p = data[0]; t_s = data[1]; b = data[2]

    # Reading parameters from configuration:
    bounds = config['grid']['bounds']; N_lon = config['grid']['N_lon']
    N_lat = config['grid']['N_lat']; N_d = config['grid']['N_d']
    t_start = date2num(UTCDateTime(config['time-grid']['t_start']).datetime)
    t_end = date2num(UTCDateTime(config['time-grid']['t_end']).datetime)
    t0s = np.linspace(t_start,t_end,config['time-grid']['N_t'])
    stla = config['stations']['stla']
    stlo = config['stations']['stlo']

    baz = []
    for b_i in config['data']['b']:
        if b_i == '':
            baz.append(None)
        else:
            baz.append(float(b_i))
    
    lons = np.linspace(bounds[0], bounds[1], N_lon)
    lats = np.linspace(bounds[2], bounds[3], N_lat)
    x, y = np.meshgrid(lons, lats)
    z = np.linspace(bounds[4], bounds[5], N_d)

    # ---------
    fig = plt.figure(figsize=(8, 6))
    ax2 = plt.subplot2grid((4, 3), (1, 0), rowspan=2, colspan=2, projection=ccrs.PlateCarree())
    # ---------

    # Numerical integration origin times and depths (computing 2D marginal distribution):
    likl_orig = likl.copy()
    likl = np.squeeze(likl_orig)
    if len(likl.shape) == 4:
        # likl has dims [times, lons, lats, depths]
        likl = simps(likl, t0s, axis=0)
        likl = simps(likl, z, axis=2)
    elif len(likl.shape) == 3:
        # likl has dims [times, lons, lats]
        likl = simps(likl, t0s, axis=0)
    else:
        return None
    
    # Normalizing the marginal distribution to unity:
    likl = likl / simps(simps(likl, lats), lons)

    if do_smoothing:
        likl = gaussian_filter(likl, smoothing_width)
    
    i_, j_ = np.unravel_index(likl.argmax(), likl.shape)

    #fig = plt.figure(figsize=(8, 6))

    #ax2 = plt.axes(projection=ccrs.PlateCarree())
    
    for i in range(0, len(stla)):
        ax2.plot(stlo[i], stla[i], marker='^', color='w', mec='k', mew=1,
                transform=ccrs.Geodetic())
        if plot_baz:
            if baz[i] is not None:
                lons_along_gcpath = [stlo[i]]
                lats_along_gcpath = [stla[i]]
                for j in range(1, 1000):
                    fwd = g.fwd(stlo[i], stla[i], baz[i], j * 1000)
                    lons_along_gcpath.append(fwd[0])
                    lats_along_gcpath.append(fwd[1])
                ax2.plot(lons_along_gcpath, lats_along_gcpath, linewidth=0.4, color='r', transform=ccrs.Geodetic())
    
    for i in range(0, len(stla)):
        ax2.plot(stlo[i], stla[i], marker='^', color='w', mec='k', mew=1,
                transform=ccrs.Geodetic())
    if shp_file is not None:
        shp = shapereader.Reader(shp_file)
        for record, geometry in zip(shp.records(), shp.geometries()):
            ax2.add_geometries([geometry], ccrs.PlateCarree(), facecolor='lightgray', alpha=0.3,
                            edgecolor='black')
    else:
        ax2.coastlines(resolution='10m')
    ax2.pcolormesh(x-np.diff(lons)[0]/2, y-np.diff(lats)[0]/2, likl[:,:], transform=ccrs.PlateCarree(),
                cmap=plt.get_cmap(cmap))
    if plot_credibility:
        ax2.contour(lons, lats, likl, [contour_threshold(likl, lons, lats, cred=cred)], colors=['k'], 
                    transform=ccrs.PlateCarree())
    if plot_true:
        if override_extent:
            ax2.plot([override_extent[0], override_extent[1]], [evla, evla], 'b--', lw=1,
                transform=ccrs.PlateCarree())
            ax2.plot([evlo, evlo], [override_extent[2], override_extent[3]], 'b--', lw=1,
                transform=ccrs.PlateCarree())
        else:
            ax2.plot([np.min(lons), np.max(lons)], [evla, evla], 'b--', lw=1,
                transform=ccrs.PlateCarree())
            ax2.plot([evlo, evlo], [np.min(lats), np.max(lats)], 'b--', lw=1,
                transform=ccrs.PlateCarree())
    if plot_maxlikl:
        if override_extent:
            ax2.plot([override_extent[0], override_extent[1]], [lats[i_], lats[i_]], 'r--', lw=1,
                transform=ccrs.PlateCarree())
            ax2.plot([lons[j_], lons[j_]], [override_extent[2], override_extent[3]], 'r--', lw=1,
                transform=ccrs.PlateCarree())
        else:
            ax2.plot([np.min(lons), np.max(lons)], [lats[i_], lats[i_]], 'r--', lw=1,
                transform=ccrs.PlateCarree())
            ax2.plot([lons[j_], lons[j_]], [np.min(lats), np.max(lats)], 'r--', lw=1,
                transform=ccrs.PlateCarree())
    ax2.set_extent([bounds[0], bounds[1], bounds[2], bounds[3]])
    if override_extent is not None:
        ax2.set_extent([override_extent[0], override_extent[1], override_extent[2], override_extent[3]])
    scale_bar(ax2, scale_bar_length, location=scale_bar_position)

    borders = cfeature.NaturalEarthFeature(category='cultural', name='admin_0_boundary_lines_land', scale='10m',
                                      facecolor='none', edgecolor='grey', linewidth=1)
    ax2.add_feature(borders)
    gl = ax2.gridlines(draw_labels=True)
    gl.xlabels_top = False
    gl.ylabels_right = False
    # ---------






    ax4 = plt.subplot2grid((4, 3), (3, 0), colspan=2)
    if aspect2 is not None:
        ax4.set_aspect(aspect2)
    
    likl = np.squeeze(likl_orig).copy()
    if len(likl.shape) == 4:
        # likl has dims [times, lons, lats, depths]
        likl = simps(likl, t0s, axis=0)
        likl = simps(likl, lons, axis=0)
    elif len(likl.shape) == 3:
        # likl has dims [times, lons, lats]
        likl = simps(likl, t0s, axis=0)
    else:
        return None
    
    # Normalizing the marginal distribution to unity:
    likl = likl / simps(simps(likl, z), lats)

    if do_smoothing:
        likl = gaussian_filter(likl, smoothing_width)
    
    i_, j_ = np.unravel_index(likl.argmax(), likl.shape)

    # ---------
    #pdb.set_trace()
    plt.pcolormesh(lons, -z, likl.transpose(), cmap=plt.get_cmap(cmap))
    #pdb.set_trace()
    #plt.pcolormesh(lats, z, likl[:,:], cmap=plt.get_cmap(cmap))
    plt.contour(lons, -z, likl.transpose(), [contour_threshold(likl, lats, z, cred=cred)], colors=['k'])

    if plot_true:
        ax4.plot([evlo, evlo], [np.min(-z), np.max(-z)], 'b--', lw=1)
        ax4.plot([np.min(lons), np.max(lons)], [-evdp, -evdp], 'b--', lw=1)
    
    #plt.show()
    # ---------





    ax3 = plt.subplot2grid((4, 3), (1, 2), rowspan=2)
    if aspect1 is not None:
        ax3.set_aspect(aspect1)
    
    likl = np.squeeze(likl_orig).copy()
    if len(likl.shape) == 4:
        # likl has dims [times, lons, lats, depths]
        likl = simps(likl, t0s, axis=0)
        likl = simps(likl, lats, axis=1)
    elif len(likl.shape) == 3:
        # likl has dims [times, lons, lats]
        likl = simps(likl, t0s, axis=0)
    else:
        return None
    
    # Normalizing the marginal distribution to unity:
    likl = likl / simps(simps(likl, z), lons)

    if do_smoothing:
        likl = gaussian_filter(likl, smoothing_width)
    
    i_, j_ = np.unravel_index(likl.argmax(), likl.shape)

    # ---------
    #pdb.set_trace()
    #plt.pcolormesh(lats, -z, likl.transpose(), cmap=plt.get_cmap(cmap))
    plt.pcolormesh(z, lats, likl, cmap=plt.get_cmap(cmap))
    plt.contour(z, lats, likl, [contour_threshold(likl, lons, z, cred=cred)], colors=['k'])

    if plot_true:
        ax3.plot([np.min(z), np.max(z)], [evla, evla], 'b--', lw=1)
        ax3.plot([evdp, evdp], [np.min(lats), np.max(lats)], 'b--', lw=1)
    
    #plt.show()
    # ---------
    plt.setp(ax3.get_yticklabels(), visible=False)
    plt.setp(ax4.get_xticklabels(), visible=False)
    #return likl

def plot_marginal_distribution2D_Depth(config, likl, aspect1=None, aspect2=None,
                             cmap='hot_r', scale_bar_length=5, scale_bar_position=(0.85,0.05),
                             override_extent=None, plot_maxlikl=False, plot_baz=True,
                             plot_true=False, evla=None, evlo=None, evdp=None, 
                             plot_credibility=True, cred=0.95, do_smoothing=False, smoothing_width=1):
    '''
    Plots the 2D marginal posterior distribution over location by integrating over
    origin time and depth
    '''

    g = Geod(ellps='sphere')

    # Reading data from prediction_file:
    data = pickle.load(open(config['model_file'], 'rb'))
    t_p = data[0]; t_s = data[1]; b = data[2]

    # Reading parameters from configuration:
    bounds = config['grid']['bounds']; N_lon = config['grid']['N_lon']
    N_lat = config['grid']['N_lat']; N_d = config['grid']['N_d']
    t_start = date2num(UTCDateTime(config['time-grid']['t_start']).datetime)
    t_end = date2num(UTCDateTime(config['time-grid']['t_end']).datetime)
    t0s = np.linspace(t_start,t_end,config['time-grid']['N_t'])
    stla = config['stations']['stla']
    stlo = config['stations']['stlo']

    baz = []
    for b_i in config['data']['b']:
        if b_i == '':
            baz.append(None)
        else:
            baz.append(float(b_i))
    
    lons = np.linspace(bounds[0], bounds[1], N_lon)
    lats = np.linspace(bounds[2], bounds[3], N_lat)
    x, y = np.meshgrid(lons, lats)
    z = np.linspace(bounds[4], bounds[5], N_d)

    # Numerical integration origin times and depths (computing 2D marginal distribution):
    likl = np.squeeze(likl)
    if len(likl.shape) == 4:
        # likl has dims [times, lons, lats, depths]
        likl = simps(likl, t0s, axis=0)
        likl = simps(likl, lons, axis=0)
    elif len(likl.shape) == 3:
        # likl has dims [times, lons, lats]
        likl = simps(likl, t0s, axis=0)
    else:
        return None
    
    # Normalizing the marginal distribution to unity:
    likl = likl / simps(simps(likl, z), lats)

    if do_smoothing:
        likl = gaussian_filter(likl, smoothing_width)
    
    i_, j_ = np.unravel_index(likl.argmax(), likl.shape)

    # ---------
    #pdb.set_trace()
    plt.pcolormesh(lons, -z, likl.transpose(), cmap=plt.get_cmap(cmap))
    #pdb.set_trace()
    #plt.pcolormesh(lats, z, likl[:,:], cmap=plt.get_cmap(cmap))
    plt.contour(lons, -z, likl.transpose(), [contour_threshold(likl, lats, z, cred=cred)], colors=['k'])
    plt.show()
    # ---------

    return likl

def plot_marginal_distribution2D(config, likl, aspect1=None, aspect2=None,
                             cmap='hot_r', scale_bar_length=5, scale_bar_position=(0.85,0.05),
                             override_extent=None, plot_maxlikl=False, plot_baz=True,
                             plot_true=False, evla=None, evlo=None, evdp=None, 
                             plot_credibility=True, cred=0.95, do_smoothing=False, smoothing_width=1,
                             plot_is_stations=True, plot_s_stations=True, shp_file=None):
    '''
    Plots the 2D marginal posterior distribution over location by integrating over
    origin time and depth
    '''

    g = Geod(ellps='sphere')

    # Reading data from prediction_file:
    data = pickle.load(open(config['model_file'], 'rb'))
    t_p = data[0]; t_s = data[1]; b = data[2]

    # Reading parameters from configuration:
    bounds = config['grid']['bounds']; N_lon = config['grid']['N_lon']
    N_lat = config['grid']['N_lat']; N_d = config['grid']['N_d']
    t_start = date2num(UTCDateTime(config['time-grid']['t_start']).datetime)
    t_end = date2num(UTCDateTime(config['time-grid']['t_end']).datetime)
    t0s = np.linspace(t_start,t_end,config['time-grid']['N_t'])
    stla = config['stations']['stla']
    stlo = config['stations']['stlo']

    baz = []
    for b_i in config['data']['b']:
        if b_i == '':
            baz.append(None)
        else:
            baz.append(float(b_i))
    
    #pdb.set_trace()

    lons = np.linspace(bounds[0], bounds[1], N_lon)
    lats = np.linspace(bounds[2], bounds[3], N_lat)
    x, y = np.meshgrid(lons, lats)
    z = np.linspace(bounds[4], bounds[5], N_d)

    # Numerical integration origin times and depths (computing 2D marginal distribution):
    likl = np.squeeze(likl)
    if len(likl.shape) == 4:
        # likl has dims [times, lons, lats, depths]
        likl = simps(likl, t0s, axis=0)
        likl = simps(likl, z, axis=2)
    elif len(likl.shape) == 3:
        # likl has dims [times, lons, lats]
        likl = simps(likl, t0s, axis=0)
    else:
        return None
    
    # Normalizing the marginal distribution to unity:
    likl = likl / simps(simps(likl, lats), lons)

    if do_smoothing:
        likl = gaussian_filter(likl, smoothing_width)
    
    i_, j_ = np.unravel_index(likl.argmax(), likl.shape)

    fig = plt.figure(figsize=(8, 6))

    ax2 = plt.axes(projection=ccrs.PlateCarree())
    
    if shp_file is not None:
        shp = shapereader.Reader(shp_file)
        for record, geometry in zip(shp.records(), shp.geometries()):
            ax2.add_geometries([geometry], ccrs.PlateCarree(), facecolor='lightgray', alpha=0.3,
                            edgecolor='black')
    else:
        ax2.coastlines(resolution='10m')
    
    for i in range(0, len(stla)):
        if baz[i] == None:
            if plot_s_stations:
                ax2.plot(stlo[i], stla[i], marker='^', color='k', mec='k', mew=1,
                    transform=ccrs.Geodetic())
        else:
            if plot_is_stations:
                ax2.plot(stlo[i], stla[i], marker='^', color='w', mec='k', mew=1,
                        transform=ccrs.Geodetic())
        if plot_baz:
            if baz[i] is not None:
                lons_along_gcpath = [stlo[i]]
                lats_along_gcpath = [stla[i]]
                for j in range(1, 1000):
                    fwd = g.fwd(stlo[i], stla[i], baz[i], j * 1000)
                    lons_along_gcpath.append(fwd[0])
                    lats_along_gcpath.append(fwd[1])
                ax2.plot(lons_along_gcpath, lats_along_gcpath, linewidth=0.4, color='r', transform=ccrs.Geodetic())
    
    #for i in range(0, len(stla)):
    #    ax2.plot(stlo[i], stla[i], marker='^', color='w', mec='k', mew=1,
    #            transform=ccrs.Geodetic())
    ax2.pcolormesh(x-np.diff(lons)[0]/2, y-np.diff(lats)[0]/2, likl[:,:], transform=ccrs.PlateCarree(),
                cmap=plt.get_cmap(cmap))
    if plot_credibility:
        ax2.contour(lons, lats, likl, [contour_threshold(likl, lons, lats, cred=cred)], colors=['k'], 
                    transform=ccrs.PlateCarree())
    if plot_true:
        if override_extent:
            ax2.plot([override_extent[0], override_extent[1]], [evla, evla], 'b--', lw=1,
                transform=ccrs.PlateCarree())
            ax2.plot([evlo, evlo], [override_extent[2], override_extent[3]], 'b--', lw=1,
                transform=ccrs.PlateCarree())
        else:
            ax2.plot([np.min(lons), np.max(lons)], [evla, evla], 'b--', lw=1,
                transform=ccrs.PlateCarree())
            ax2.plot([evlo, evlo], [np.min(lats), np.max(lats)], 'b--', lw=1,
                transform=ccrs.PlateCarree())
    if plot_maxlikl:
        if override_extent:
            ax2.plot([override_extent[0], override_extent[1]], [lats[i_], lats[i_]], 'r--', lw=1,
                transform=ccrs.PlateCarree())
            ax2.plot([lons[j_], lons[j_]], [override_extent[2], override_extent[3]], 'r--', lw=1,
                transform=ccrs.PlateCarree())
        else:
            ax2.plot([np.min(lons), np.max(lons)], [lats[i_], lats[i_]], 'r--', lw=1,
                transform=ccrs.PlateCarree())
            ax2.plot([lons[j_], lons[j_]], [np.min(lats), np.max(lats)], 'r--', lw=1,
                transform=ccrs.PlateCarree())
    ax2.set_extent([bounds[0], bounds[1], bounds[2], bounds[3]])
    if override_extent is not None:
        ax2.set_extent([override_extent[0], override_extent[1], override_extent[2], override_extent[3]])
    scale_bar(ax2, scale_bar_length, location=scale_bar_position)
    #ax2.coastlines(resolution='10m')
    borders = cfeature.NaturalEarthFeature(category='cultural', name='admin_0_boundary_lines_land', scale='10m',
                                      facecolor='none', edgecolor='grey', linewidth=1)
    ax2.add_feature(borders)
    gl = ax2.gridlines(draw_labels=True)
    gl.xlabels_top = False
    gl.ylabels_right = False

    return likl

def plot_likelihood_function(config, likl, plot_true=False,
                             plot_maxlikl=True, evla=None, evlo=None, evdp=None, evt0=None, 
                             aspect0=None, aspect1=None, aspect2=None,
                             cmap='bone_r', scale_bar_length=5, scale_bar_position=(0.85,0.05),
                             override_extent=None, baz=None, plot_times=False, plot_depth_slices=False):
    '''
    Plots the 4D likelihood function, maximum likelihood value, and optionally (if plot_true=True)
    the true or known location

    Note that aspect has to be set manually by trial and error to ensure that the subplots line
    up correctly

    Use override_extent to define a custom lat/lon boundary: override_extent = [lon0,lon1,lat0,lat1]

    If using only backazimuths, set plot_all_subplots = False to just show a map
    '''

    g = Geod(ellps='sphere')

    # Reading data from prediction_file:
    data = pickle.load(open(config['model_file'], 'rb'))
    t_p = data[0]; t_s = data[1]; b = data[2]

    # Reading parameters from configuration:
    bounds = config['grid']['bounds']; N_lon = config['grid']['N_lon']
    N_lat = config['grid']['N_lat']; N_d = config['grid']['N_d']
    t_start = date2num(UTCDateTime(config['time-grid']['t_start']).datetime)
    t_end = date2num(UTCDateTime(config['time-grid']['t_end']).datetime)
    t0s = np.linspace(t_start,t_end,config['time-grid']['N_t'])
    stla = config['stations']['stla']
    stlo = config['stations']['stlo']

    lons = np.linspace(bounds[0], bounds[1], N_lon)
    lats = np.linspace(bounds[2], bounds[3], N_lat)
    x, y = np.meshgrid(lons, lats)
    z = np.linspace(bounds[4], bounds[5], N_d)

    k_, i_, j_, l_ = np.unravel_index(likl.argmax(), likl.shape)

    fig = plt.figure(figsize=(8, 6))

    if plot_times:
        ax1 = plt.subplot2grid((4, 3), (0, 0), colspan=3)
        plt.plot_date(t0s, likl[:, i_, j_, l_], 'k-')
        if aspect0 is not None:
            ax1.set_aspect(aspect0)
        y_lims = plt.gca().get_ylim()
        if plot_true:
            plt.plot_date([evt0, evt0], [y_lims[0], y_lims[1]], 'b--')
        if plot_maxlikl:
            plt.plot_date([t0s[k_], t0s[k_]], [y_lims[0], y_lims[1]], 'r--')
        plt.ylim(y_lims)
        plt.gca().xaxis.tick_top()
        plt.gca().set_yticks([])
    
    if plot_times:
        if plot_depth_slices:
            ax2 = plt.subplot2grid((4, 3), (1, 0), rowspan=2, colspan=2, projection=ccrs.PlateCarree())
        else:
            ax2 = plt.subplot2grid((4, 3), (1, 0), rowspan=3, colspan=3, projection=ccrs.PlateCarree())
    else:
        ax2 = plt.axes(projection=ccrs.PlateCarree())
    
    for i in range(0, len(stla)):
        ax2.plot(stlo[i], stla[i], marker='^', color='w', mec='k', mew=1,
                transform=ccrs.Geodetic())
        if baz is not None:
            if baz[i] is not None:
                lons_along_gcpath = [stlo[i]]
                lats_along_gcpath = [stla[i]]
                for j in range(1, 1000):
                    fwd = g.fwd(stlo[i], stla[i], baz[i], j * 1000)
                    lons_along_gcpath.append(fwd[0])
                    lats_along_gcpath.append(fwd[1])
                ax2.plot(lons_along_gcpath, lats_along_gcpath, linewidth=0.4, color='r', transform=ccrs.Geodetic())
    ax2.pcolormesh(x-np.diff(lons)[0]/2, y-np.diff(lats)[0]/2, likl[k_,:,:,l_], transform=ccrs.PlateCarree(),
                cmap=plt.get_cmap(cmap))
    if plot_true:
        if override_extent:
            ax2.plot([override_extent[0], override_extent[1]], [evla, evla], 'b--', lw=1,
                transform=ccrs.PlateCarree())
            ax2.plot([evlo, evlo], [override_extent[2], override_extent[3]], 'b--', lw=1,
                transform=ccrs.PlateCarree())
        else:
            ax2.plot([np.min(lons), np.max(lons)], [evla, evla], 'b--', lw=1,
                transform=ccrs.PlateCarree())
            ax2.plot([evlo, evlo], [np.min(lats), np.max(lats)], 'b--', lw=1,
                transform=ccrs.PlateCarree())
    if plot_maxlikl:
        if override_extent:
            ax2.plot([override_extent[0], override_extent[1]], [lats[i_], lats[i_]], 'r--', lw=1,
                transform=ccrs.PlateCarree())
            ax2.plot([lons[j_], lons[j_]], [override_extent[2], override_extent[3]], 'r--', lw=1,
                transform=ccrs.PlateCarree())
        else:
            ax2.plot([np.min(lons), np.max(lons)], [lats[i_], lats[i_]], 'r--', lw=1,
                transform=ccrs.PlateCarree())
            ax2.plot([lons[j_], lons[j_]], [np.min(lats), np.max(lats)], 'r--', lw=1,
                transform=ccrs.PlateCarree())
    ax2.set_extent([bounds[0], bounds[1], bounds[2], bounds[3]])
    if override_extent is not None:
        ax2.set_extent([override_extent[0], override_extent[1], override_extent[2], override_extent[3]])
    scale_bar(ax2, scale_bar_length, location=scale_bar_position)
    
    ax2.coastlines(resolution='10m')
    borders = cfeature.NaturalEarthFeature(category='cultural', name='admin_0_boundary_lines_land', scale='10m',
                                      facecolor='none', edgecolor='grey', linewidth=1)
    ax2.add_feature(borders)
    gl = ax2.gridlines(draw_labels=True)
    gl.xlabels_top = False
    gl.ylabels_right = False

    if plot_depth_slices:
        ax3 = plt.subplot2grid((4, 3), (1, 2), rowspan=2)
        plt.pcolormesh(z-np.diff(z)[0]/2, lats-np.diff(lats)[0]/2, likl[k_,:,j_,:], cmap=plt.get_cmap(cmap))
        if aspect1 is not None:
            ax3.set_aspect(aspect1)                                # This has to be found manually by trial and error!
        if plot_true:
            if override_extent:
                ax3.plot([np.min(z), np.max(z)], [evla, evla], 'b--', lw=1)
                ax3.plot([evdp, evdp], [override_extent[2], override_extent[3]], 'b--', lw=1)
            else:
                ax3.plot([np.min(z), np.max(z)], [evla, evla], 'b--', lw=1)
                ax3.plot([evdp, evdp], [np.min(lats), np.max(lats)], 'b--', lw=1)
        if plot_maxlikl:
            if override_extent:
                ax3.plot([np.min(z), np.max(z)], [lats[i_], lats[i_]], 'r--', lw=1)
                ax3.plot([z[l_], z[l_]], [override_extent[2], override_extent[3]], 'r--', lw=1)
            else:
                ax3.plot([np.min(z), np.max(z)], [lats[i_], lats[i_]], 'r--', lw=1)
                ax3.plot([z[l_], z[l_]], [np.min(lats), np.max(lats)], 'r--', lw=1)
        plt.xlim([bounds[4], bounds[5]])
        plt.ylim([bounds[2], bounds[3]])
        if override_extent is not None:
            plt.ylim([override_extent[2], override_extent[3]])
        plt.xlabel('Depth (km)')

    if plot_depth_slices:
        ax4 = plt.subplot2grid((4, 3), (3, 0), colspan=2)
        plt.pcolormesh(lons-np.diff(lons)[0]/2, z-np.diff(z)[0]/2, likl[k_,i_,:,:].transpose(), cmap=plt.get_cmap(cmap))
        plt.gca().xaxis.tick_top()
        if aspect2 is not None:
            ax4.set_aspect(aspect2)
        if plot_true:
            if override_extent:
                ax4.plot([evlo, evlo], [np.min(z), np.max(z)], 'b--', lw=1)
                ax4.plot([override_extent[0], override_extent[1]], [evdp, evdp], 'b--', lw=1)
            else:
                ax4.plot([evlo, evlo], [np.min(z), np.max(z)], 'b--', lw=1)
                ax4.plot([np.min(lons), np.max(lons)], [evdp, evdp], 'b--', lw=1)
        if plot_maxlikl:
            if override_extent:
                ax4.plot([lons[j_], lons[j_]], [np.min(z), np.max(z)], 'r--', lw=1)
                ax4.plot([override_extent[0], override_extent[1]], [z[l_], z[l_]], 'r--', lw=1)
            else:
                ax4.plot([lons[j_], lons[j_]], [np.min(z), np.max(z)], 'r--', lw=1)
                ax4.plot([np.min(lons), np.max(lons)], [z[l_], z[l_]], 'r--', lw=1)
        plt.xlim([bounds[0], bounds[1]])
        if override_extent is not None:
            plt.xlim([override_extent[0], override_extent[1]])
        plt.ylim([bounds[4], bounds[5]])
        plt.ylabel('Depth (km)')
        plt.gca().invert_yaxis()

        plt.setp(ax3.get_yticklabels(), visible=False)
        plt.setp(ax4.get_xticklabels(), visible=False)

def scale_bar(ax, length=None, location=(0.5, 0.05), linewidth=3):
    """
    ax is the axes to draw the scalebar on.
    length is the length of the scalebar in km.
    location is center of the scalebar in axis coordinates.
    (ie. 0.5 is the middle of the plot)
    linewidth is the thickness of the scalebar.
    """
    #Get the limits of the axis in lat long
    llx0, llx1, lly0, lly1 = ax.get_extent(ccrs.PlateCarree())
    #Make tmc horizontally centred on the middle of the map,
    #vertically at scale bar location
    sbllx = (llx1 + llx0) / 2
    sblly = lly0 + (lly1 - lly0) * location[1]
    tmc = ccrs.TransverseMercator(sbllx, sblly)
    #Get the extent of the plotted area in coordinates in metres
    x0, x1, y0, y1 = ax.get_extent(tmc)
    #Turn the specified scalebar location into coordinates in metres
    sbx = x0 + (x1 - x0) * location[0]
    sby = y0 + (y1 - y0) * location[1]

    #Calculate a scale bar length if none has been given
    #(Theres probably a more pythonic way of rounding the number but this works)
    if not length:
        length = (x1 - x0) / 5000 #in km
        ndim = int(np.floor(np.log10(length))) #number of digits in number
        length = round(length, -ndim) #round to 1sf
        #Returns numbers starting with the list
        def scale_number(x):
            if str(x)[0] in ['1', '2', '5']: return int(x)
            else: return scale_number(x - 10 ** ndim)
        length = scale_number(length)

    #Generate the x coordinate for the ends of the scalebar
    bar_xs = [sbx - length * 500, sbx + length * 500]
    #Plot the scalebar
    ax.plot(bar_xs, [sby, sby], transform=tmc, color='k', linewidth=linewidth)
    #Plot the scalebar label
    ax.text(sbx, sby, str(length) + ' km', transform=tmc,
            horizontalalignment='center', verticalalignment='bottom')
