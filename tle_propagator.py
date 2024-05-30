'''
imports a TLE from spacetrack
uses orekit to add perturbations and propagate the orbit
records positions throughout propagation
saves positions to a txt file
'''

#Imports
import orekit
from orekit import JArray_double

from orekit.pyhelpers import setup_orekit_curdir, absolutedate_to_datetime
from org.orekit.frames import FramesFactory, TopocentricFrame
from org.orekit.orbits import OrbitType
from org.orekit.time import TimeScalesFactory, AbsoluteDate
from org.orekit.utils import Constants, IERSConventions, PVCoordinatesProvider, PVCoordinates
from org.orekit.bodies import OneAxisEllipsoid, GeodeticPoint
from org.orekit.estimation.measurements import GroundStation
from org.orekit.bodies import CelestialBodyFactory
from org.orekit.propagation.analytical.tle import TLE, TLEPropagator
from org.orekit.utils import PVCoordinatesProvider
from org.orekit.propagation.numerical import NumericalPropagator
from org.hipparchus.ode.nonstiff import DormandPrince853Integrator
from org.orekit.propagation import SpacecraftState

# For setting up new tle propagator
from org.orekit.time import FixedStepSelector
from org.orekit.estimation.measurements import AngularAzEl
from org.orekit.estimation.measurements.generation import Generator, AngularAzElBuilder, EventBasedScheduler, SignSemantic, GatheringSubscriber
from org.orekit.geometry.fov import DoubleDihedraFieldOfView, CircularFieldOfView
from org.orekit.propagation.events import GroundFieldOfViewDetector, GroundAtNightDetector, EclipseDetector, BooleanDetector

# For setting up earth
from org.orekit.bodies import  OneAxisEllipsoid, GeodeticPoint

# For setting up perturbations
from org.orekit.models.earth.atmosphere.data import MarshallSolarActivityFutureEstimation
from org.orekit.forces.radiation import SolarRadiationPressure, IsotropicRadiationSingleCoefficient
from org.orekit.forces.gravity import ThirdBodyAttraction, HolmesFeatherstoneAttractionModel, Relativity
from org.orekit.forces.gravity.potential import GravityFieldFactory
from org.orekit.models.earth.atmosphere import NRLMSISE00
from org.orekit.forces.drag import IsotropicDrag
from org.orekit.forces.drag import DragForce

#for Event Handler
from org.hipparchus.geometry.euclidean.threed import Vector3D
from org.hipparchus.geometry import Vector
from org.orekit.propagation.events import EventsLogger
from org.orekit.propagation.events.handlers import ContinueOnEvent


#for field of view detector
from org.orekit.time import AbsoluteDate
from org.orekit.utils import Constants
from org.orekit.propagation.events.handlers import ContinueOnEvent

from math import radians, degrees, atan, tan, sqrt, pi, sin, cos, acos
import argparse
import random
import pandas as pd
import sys

#Initialize orekit
vm = orekit.initVM()
setup_orekit_curdir("./orekit-data-master/")

class space_object_chars:
	def __init__(self, shape, mass, cross_section, cd, cr):
		self.mass           = mass
		self.cross_section  = cross_section
		self.cd             = cd
		self.cr             = cr
		self.shape          = shape

def get_tle_lines(file):
    # Reads tle lines for files that only have sat name, line1, and line2
    with open(file, "r") as file:
        # Read the TLE data
        tle_data = file.readlines()

    # Process the TLE data
    satellite_name = tle_data[0].strip()  # Strip any leading/trailing whitespace
    line1 = tle_data[1].strip()
    line2 = tle_data[2].strip()


    return satellite_name, line1, line2

def get_tle_info(file, print_info=False):
    tle_info = {}
    with open(file, "r") as file:
        for line in file:
            key, value = line.strip().split(": ")
            tle_info[key] = value
    
    a = float(tle_info['SEMIMAJOR_AXIS'])  # Semi-major axis (m)
    e = float(tle_info['ECCENTRICITY'])     # Eccentricity
    i = radians(float(tle_info['INCLINATION']))      # Inclination 
    omega = radians(float(tle_info['ARG_OF_PERICENTER']))  # Argument of perigee 
    raan = radians(float(tle_info['RA_OF_ASC_NODE']))   # Right ascension of ascending node 
    mean_anom = radians(float(tle_info['MEAN_ANOMALY']))      # Mean anomaly 
    ra = float(tle_info['APOGEE']) # Apogee
    rp = float(tle_info['PERIGEE']) # Perigee
    theta = mean_to_true_anom(mean_anom, e) # True anomaly
    line0 = tle_info['TLE_LINE0']
    line1 = tle_info['TLE_LINE1']
    line2 = tle_info['TLE_LINE2']

    t0_y = int(tle_info['START_Y'])
    t0_m = int(tle_info['START_M'])
    t0_d = int(tle_info['START_D'])
    t0_hr = int(tle_info['START_HR'])
    t0_min = int(tle_info['START_MIN'])
    t0_sec = float(tle_info['START_SEC'])
    t1_y = int(tle_info['END_Y'])
    t1_m = int(tle_info['END_M'])
    t1_d = int(tle_info['END_D'])
    t1_hr = int(tle_info['END_HR'])
    t1_min = int(tle_info['END_MIN'])
    t1_sec = float(tle_info['END_SEC'])

    t0 = AbsoluteDate(t0_y,t0_m,t0_d,t0_hr,t0_min,t0_sec, TimeScalesFactory.getUTC())
    t1 = AbsoluteDate(t1_y,t1_m,t1_d,t1_hr,t1_min,t1_sec, TimeScalesFactory.getUTC())

    if print_info:
        # Print orbit information
        print('TLE Epoch:', tle_epoch)
        print('Semi-major axis:', a)
        print('Eccentricity:', e)
        print('Inclination:', i)
        print('Argument of Perigee:', omega)
        print('Right Ascension of Ascending Node:', raan)
        print('True Anomaly:', theta)
        print('Apogee:', ra)
        print('Perigee:', rp)
    
    return line0, line1, line2, t0, t1

def mean_to_true_anom(Me, e):
    #mean anomaly should already be in radians

    #Use algorithm to get the eccentricc anomaly from the mena anomaly
    thresh = 1e-10
    # iterations = 0
    if Me < pi:
        E = Me + e/2
    else:
        E = Me - e/2
    f = E - e*sin(E) - Me
    f_prime = 1 - e*cos(E)

    while abs(f/f_prime) > thresh:
        # iterations += 1
        E = E - f/f_prime
        f = E - e*sin(E) - Me
        f_prime = 1 - e*cos(E)

    return 2*atan(sqrt((1+e)/(1-e))*tan(E/2))

def sphere2cartesian(phi, theta, r=1):
    phi = radians(90.0 - phi)
    theta = radians(theta)
    
    return [
        r * cos(phi) * cos(theta),
        r * sin(phi) * cos(theta),
        r * sin(theta)
    ]

def addForceModels(propagator, space_object_chars, relativity=True, thirdBody=True, solarPressure=True, atmDrag=True):
    sun = CelestialBodyFactory.getSun()
    moon = CelestialBodyFactory.getMoon()
    gravityProvider = GravityFieldFactory.getNormalizedProvider(70, 70)
    propagator.addForceModel(HolmesFeatherstoneAttractionModel(earth.getBodyFrame(), gravityProvider))

    if (relativity):
        relativityEffect = Relativity(Constants.WGS84_EARTH_MU)
        propagator.addForceModel(relativityEffect)
    
    if (thirdBody):
        propagator.addForceModel(ThirdBodyAttraction(sun))
        propagator.addForceModel(ThirdBodyAttraction(moon))

    if (solarPressure):
        isotropicRadiationSingleCoeff = IsotropicRadiationSingleCoefficient(space_object_chars.cross_section, space_object_chars.cr)
        solarRadiationPressure = SolarRadiationPressure(sun, 
                                                        earth,
			                                            isotropicRadiationSingleCoeff)
        propagator.addForceModel(solarRadiationPressure)
    
    if (atmDrag):
        parameters = MarshallSolarActivityFutureEstimation(MarshallSolarActivityFutureEstimation.DEFAULT_SUPPORTED_NAMES,
                                                           MarshallSolarActivityFutureEstimation.StrengthLevel.AVERAGE)
        atmosphere = NRLMSISE00(parameters, sun, earth)
        isotropicDrag = IsotropicDrag(space_object_chars.cross_section, space_object_chars.cd)
        dragForce = DragForce(atmosphere, isotropicDrag)
        propagator.addForceModel(dragForce)

    return propagator

def createDihedralFOV(az, el):
    x,y,z = sphere2cartesian(az, el)
    center = Vector3D([x,y,z])
    axis2 = Vector3D.crossProduct(Vector3D.PLUS_K, center)
    axis1 = Vector3D.crossProduct(axis2, center)
    Vector.cast_(axis1).normalize()
    Vector.cast_(axis2).normalize()
    ha1 = radians(fov_width)/2
    ha2 = radians(fov_height)/2
    ddfov = DoubleDihedraFieldOfView(center, axis1, ha1, axis2, ha2, 0.0)

    return ddfov

def getPositionCoord(pv, scale):
    x = pv.getPosition().getX()*scale
    y = pv.getPosition().getY()*scale
    z = pv.getPosition().getZ()*scale

    return '(%f, %f, %f)' % (x, y, z)

def getPositionCoord_gs(gs1_frame, epoch, inertial_frame, itrf_frame, scale):
    
    itrf_to_gcrf = itrf_frame.getTransformTo(inertial_frame, epoch)
    itrf_pv = gs1_frame.getPVCoordinates(epoch, itrf_frame)
    inertial_pv = itrf_to_gcrf.transformPVCoordinates(itrf_pv)
    # print('Inertial Coord: ', inertial_pv)

    x = inertial_pv.getPosition().getX()*scale
    y = inertial_pv.getPosition().getY()*scale
    z = inertial_pv.getPosition().getZ()*scale

    return '(%f, %f, %f)' % (x, y, z)

def getZenithFromGS(gs1_frame, epoch, inertial_frame, scale):
    #Greate transformation
    topo_to_gcrf = gs1_frame.getTransformTo(inertial_frame, epoch)
    
    #Transform the origin of the topocentric frame to inertial frame
    gs1_inertial_pv = gs1_frame.getPVCoordinates(event_epoch, inertial_frame)
    origin_inertial_pos = gs1_inertial_pv.getPosition()

    #Transform a point along the zenith to inertial frame
    zenith_topo = Vector3D(0.0, 0.0, 1.0)
    zenith_topo_pv = PVCoordinates(zenith_topo, Vector3D.ZERO, Vector3D.ZERO)
    zenithPt_inertial_pos = topo_to_gcrf.transformPVCoordinates(zenith_topo_pv).getPosition()

    #Calculate the zenith unit vector in inertial frame
    unitV_zenith_inertial = origin_inertial_pos.subtract(zenithPt_inertial_pos)

    #Normalize the vector - technically it should already be normalized
    norm_unitV = Vector.cast_(unitV_zenith_inertial).normalize()

    # Create the position for reference object
    ref_pv = PVCoordinates(Vector3D(0.0, 0.0, 10000000.0), Vector3D.ZERO, Vector3D.ZERO) #10,000km above the ground station
    ref_inertial_pos = topo_to_gcrf.transformPVCoordinates(ref_pv).getPosition()
    ref_x = ref_inertial_pos.getX()*scale
    ref_y = ref_inertial_pos.getY()*scale
    ref_z = ref_inertial_pos.getZ()*scale

    return norm_unitV, str(f'(%f, %f, %f)' % (ref_x, ref_y, ref_z))

def get_dot(gs1_frame, event_epoch, inertial_frame, pv_sun, surface_norm_unitV):

    # Create the position for the sun in inertial frame
    sun_pos_vector = pv_sun.getPosition()
    
    # Transform the origin of the topocentric frame to inertial frame
    gs1_inertial_pv = gs1_frame.getPVCoordinates(event_epoch, inertial_frame)
    origin_inertial_pos = gs1_inertial_pv.getPosition()

    #Calculate the zenith unit vector in inertial frame
    origin_to_sun = sun_pos_vector.subtract(origin_inertial_pos)

    #Normalize the vector - technically it should already be normalized
    unitV_topo2sun = Vector.cast_(origin_to_sun).normalize()

    #Calculate the dot product
    dot = Vector3D.dotProduct(Vector3D.cast_(unitV_topo2sun), surface_norm_unitV)

    return unitV_topo2sun, dot

def get_range(pv_obj):
    x = pv_obj.getPosition().getX()
    y = pv_obj.getPosition().getY()
    z = pv_obj.getPosition().getZ()

    return sqrt(x**2 + y**2 + z**2)

def get_phase_angle(obj_pv, sun_pv, obj_range, sun_range):
    obj_pos = obj_pv.getPosition()
    sun_pos = sun_pv.getPosition()

    dot = Vector3D.dotProduct(obj_pos, sun_pos)

    phase_angle = degrees(acos(dot/(obj_range*sun_range)))

    return phase_angle

def parse_args():
    parser = argparse.ArgumentParser(description='Process satellite parameters for TLE propagation.')
    parser.add_argument('--track_num', required=True, help='Track number')
    parser.add_argument('--track_dir', required=True, help='Directory to save tracking data')
    parser.add_argument('--obj_name', required=True, help='Name of the object')
    parser.add_argument('--obj_path', required=True, help='Path to the object stl file')
    parser.add_argument('--tle_file', required=True, help='Path to the TLE file')
    parser.add_argument('--positions_file', required=True, help='Path to save object positions')
    parser.add_argument('--meta_file', required=True, help='Path to save metadata')
    parser.add_argument('--topo_data_file', required=True, help='Path to save topocentric data')
    parser.add_argument('--scale', required=True, type=float, help='Scale to convert to blender units')
    parser.add_argument('--regime', required=True, help='Attitude regime of the satellite')
    parser.add_argument('--spin_x', required=True, type=float, help='x-axis spin rate in rad/s')
    parser.add_argument('--spin_y', required=True, type=float, help='y-axis spin rate in rad/s')
    parser.add_argument('--spin_z', required=True, type=float, help='z-axis spin rate in rad/s')
    parser.add_argument('--mass', required=True, type=float, help='Mass of the satellite in kg')
    parser.add_argument('--cross_sect', required=True, type=float, help='Cross section of the satellite in m^2')
    parser.add_argument('--cd', required=True, type=float, help='Coefficient of drag')
    parser.add_argument('--cr', required=True, type=float, help='Coefficient of reflectivity')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Read in Arguments ---------------------------------
    track_num = args.track_num
    track_dir = args.track_dir
    obj_name = args.obj_name
    obj_path = args.obj_path
    tle_file = args.tle_file
    positions_file = args.positions_file
    meta_file = args.meta_file
    topo_data_file = args.topo_data_file
    scale = args.scale
    regime = args.regime
    spin_x = args.spin_x
    spin_y = args.spin_y
    spin_z = args.spin_z
    mass = args.mass
    cross_sect = args.cross_sect
    cd = args.cd
    cr = args.cr
    
    print('************************************')
    print('ARGUMENTS')
    print('Track Directory:', track_dir)
    print('Object Name:', obj_name)
    print('Object Path:', obj_path)
    print('TLE File:', tle_file)
    print('Positions File:', positions_file)
    print('Meta File:', meta_file)
    print('Topographic Data File:', topo_data_file)
    print('Scale:', scale)
    print('Attitude Regime:', regime)
    print('Spin Rates:', spin_x, spin_y, spin_z)
    print('Mass:', mass)
    print('Cross Section:', cross_sect)
    print('Coefficient of Drag:', cd)
    print('Coefficient of Reflectivity:', cr)
    print('************************************')

    # Extra parameters
    obj_elev_min = radians(15)  # Elevation limit in radians
    sun_elev_max = radians(-6)
    prop_time_hrs = 0.5  # Time to propagate orbit in hours
    latitude = radians(43.6)  # Latitude of MiniMegaTORTORA telescope
    longitude = radians(41.4) # Longitude of MiniMegaTORTORA telescope
    altitude = 2030.0  # Altitude above sea level in meters
    half_ap_fov = radians(45)  # Half of the aperture field of view in radians
    minStep = 1e-3
    maxstep = 1e3
    initStep = 60.0 #[s]
    # az = 180 # azimuth
    # el = 45 # elevation
    # fov_width = radians(30)  #11 #degrees
    # fov_height = radians(27) #9  #degrees  

    #Constants -----------------------------------------------
    utc = TimeScalesFactory.getUTC()
    ae = Constants.EIGEN5C_EARTH_EQUATORIAL_RADIUS
    mu = Constants.WGS84_EARTH_MU
    inertial_frame = FramesFactory.getGCRF() #FramesFactory.getEME2000()
    itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, True) # International Terrestrial Reference Frame, earth fixed
    sun = PVCoordinatesProvider.cast_(CelestialBodyFactory.getSun())
    timestep = 0.1 # timestep in seconds for scheduler - must be float
    # ------------------------------------------------------------------------

    # Set up Space Object
    so = space_object_chars(obj_name, mass, cross_sect, cd, cr)

    #Step 2: Import the selected TLE info from file
    tle_line0, tle_line1, tle_line2, t0, t1 = get_tle_info(tle_file)
    tle = TLE(tle_line1, tle_line2)
    tle_epoch = tle.getDate() #Date of TLE epoch
    print('TLE EPOCH: ', tle_epoch)
    print('TLE:')
    print(tle_line1)
    print(tle_line2)
    print('Propagation Start Epoch: ', t0)
    print('Propagation End Epoch: ', t1)
    
    #Step 3: Set up orekit 
    # Define Earth
    itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, True) # International Terrestrial Reference Frame, earth fixed
    earth = OneAxisEllipsoid(Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
                            Constants.WGS84_EARTH_FLATTENING,
                            itrf)

    # Set up ground station
    gs1 = GeodeticPoint(latitude, longitude, altitude)
    gs1_frame = TopocentricFrame(earth, gs1, "station")
    groundStation = GroundStation(gs1_frame)
    groundStationPosition = earth.transform(gs1)

    # Set up TLE propagator to get initial orbit
    propagator_tle = TLEPropagator.selectExtrapolator(tle)
    # Get the orbit from the TLE
    orbit = propagator_tle.getInitialState().getOrbit()

    # Set up numerical propagator
    positionTolerance = 1.0 
    orbitType = OrbitType.CIRCULAR
    tol = NumericalPropagator.tolerances(positionTolerance, orbit, orbitType)
    initial_state = SpacecraftState(orbit, mass)

    integrator = DormandPrince853Integrator(minStep, maxstep, 
        JArray_double.cast_(tol[0]),  # Double array of doubles needs to be casted in Python
        JArray_double.cast_(tol[1]))
    integrator.setInitialStepSize(initStep)

    propagator = NumericalPropagator(integrator)
    propagator.setOrbitType(orbitType)
    propagator.setInitialState(initial_state)
    propagator = addForceModels(propagator, so)


    # Set up ground FoV detector
    generator = Generator()
    space_object = generator.addPropagator(propagator)
    azElBuilder = AngularAzElBuilder(None, groundStation, [0.0,0.0], [1.0,1.0], space_object)
    fov = CircularFieldOfView(Vector3D.PLUS_K, half_ap_fov, 0.0)
    fovGroundDetector = GroundFieldOfViewDetector(gs1_frame, fov).withHandler(ContinueOnEvent()).withMaxCheck(0.1) #Check every second

    # Set up night detector
    nightDetector = GroundAtNightDetector(gs1_frame, 
                                        sun, 
                                        GroundAtNightDetector.CIVIL_DAWN_DUSK_ELEVATION, 
                                        None).withHandler(ContinueOnEvent())

    # Set up eclipse detector
    eclipseDetector = EclipseDetector(sun, Constants.SUN_RADIUS, earth).withPenumbra().withHandler(ContinueOnEvent())

    # Combining detectors
    fovAtNightDetector = BooleanDetector.andCombine([BooleanDetector.notCombine(fovGroundDetector), 
                                                    nightDetector,
                                                    eclipseDetector]).withHandler(ContinueOnEvent())


    # Scheduler
    scheduler = EventBasedScheduler(azElBuilder, 
                                    FixedStepSelector(timestep, TimeScalesFactory.getUTC()), 
                                    generator.getPropagator(space_object),
                                    fovAtNightDetector,
                                    SignSemantic.FEASIBLE_MEASUREMENT_WHEN_POSITIVE)

    generator.addScheduler(scheduler)
    subscriber = GatheringSubscriber()
    generator.addSubscriber(subscriber)

    logger = EventsLogger()
    propagator.addEventDetector(logger.monitorDetector(fovAtNightDetector))

    # Retreive measurements
    generator.generate(t0, t1)
    data = subscriber.getGeneratedMeasurements()

    # Create Events Logger checks when object enters and exits field of view
    mylog = logger.getLoggedEvents()
    print('Logged Events: ', mylog.size())
    for event in mylog:
        print(event.getState().getDate())

    ## Record results
    obj_pos_list = []
    topo_data_list = []
    # sun_pos_list = []
    first = True

    for meas in data:
        castedMeasurements = AngularAzEl.cast_(meas)
        azimuth = degrees(castedMeasurements.getObservedValue()[0])
        elevation = degrees(castedMeasurements.getObservedValue()[1])
        event_epoch = castedMeasurements.getDate()

        # Calculate ranges
        pv_object_topo = propagator.getPVCoordinates(event_epoch, gs1_frame)
        pv_sun_topo = sun.getPVCoordinates(event_epoch, gs1_frame)
        range_obj = get_range(pv_object_topo)
        range_sun = get_range(pv_sun_topo)
        phase_angle = get_phase_angle(pv_object_topo, pv_sun_topo, range_obj, range_sun)
        topo_data = str(f'Epoch: {event_epoch}, Azimuth: {azimuth}, Elevation: {elevation}, Range: {range_obj}, Phase: {phase_angle}')

        pv_object = propagator.getPVCoordinates(event_epoch, inertial_frame)
        pv_sun = sun.getPVCoordinates(event_epoch, inertial_frame)

        coord_obj = getPositionCoord(pv_object, scale)
        coord_sun = getPositionCoord(pv_sun, scale)

        if first:
            # print(topo_data)
            pv_sun = sun.getPVCoordinates(event_epoch, inertial_frame)
            coord_sun = getPositionCoord(pv_sun, scale)
            
            # Get ground station position (camera position, inertial frame)
            coord_gs1 = getPositionCoord_gs(gs1_frame, event_epoch, inertial_frame, itrf, scale)

            # Get unit vector pointing to zenith from ground station and position of ref object, inertial frame
            zenith_unitVec, ref_pos = getZenithFromGS(gs1_frame, event_epoch, inertial_frame, scale)

            # Get unitV from gs1 to sun and u_sun dot product with ref surface normal vector
            u_sun, norm_dot_sun = get_dot(gs1_frame, event_epoch, inertial_frame, pv_sun, zenith_unitVec.negate())

            first = False
        
        obj_pos_list.append(coord_obj)
        topo_data_list.append(topo_data)
        # sun_pos_list.append(coord_sun)

        # ****************Calculate phase angle here****************


    print('Measurements: ', data.size())

    with open(positions_file, "w") as file:
        for item in obj_pos_list:
            file.write("%s\n"% str(item)) 

    with open(meta_file, 'w') as file:
        file.write(f'Object Name: {obj_name}\n')
        file.write(f'Attitude Regime: {regime}\n')
        file.write(f'Spin Rates [rad/s]: x: {spin_x}, y: {spin_y}, z: {spin_z}\n')
        file.write(f'Object Path: {obj_path}\n')
        file.write(f'Mass: {mass}\n')
        file.write(f'Cross Section: {cross_sect}\n')
        file.write(f'Coefficient of Drag: {cd}\n')
        file.write(f'Coefficient of Reflectivity: {cr}\n')
        file.write(f'TLE Epoch: {tle_epoch}\n')
        file.write(f'Ground Station Position: {coord_gs1}\n')
        file.write(f'Sun Position: {coord_sun}\n')
        file.write(f'Zenith Unit Vector: {zenith_unitVec}\n')
        file.write(f'Reference Position: {ref_pos}\n')
        file.write(f'Unit Vector from Topocentric Origin to Sun (u_sun): {u_sun}\n')
        file.write(f'Dot Product Betweeen u_sun and u_surfaceNorm: {norm_dot_sun}\n')
        file.write(f'TLE Line 0: {tle_line0}\n')
        file.write(f'TLE Line 1: {tle_line1}\n')
        file.write(f'TLE Line 2: {tle_line2}\n')
        file.write(f'Propagation Start Epoch: {t0}\n')
        file.write(f'Propagation End Epoch: {t1}\n')

    with open(topo_data_file, "w") as file:
        file.write('Epoch Azimuth[deg] Elevation[deg] Range[m] Phase[deg]\n')
        for item in topo_data_list:
            info = item.split(', ')
            epoch = str(info[0].split(': ')[1])
            azimuth = float(info[1].split(': ')[1])
            elevation = float(info[2].split(': ')[1])
            range = float(info[3].split(': ')[1])
            phase = float(info[4].split(': ')[1])
            file.write("%s %s %s %s %s\n"% (epoch, azimuth, elevation, range, phase))

    print(f'*** Positions Generated for Track {track_num}. ***')