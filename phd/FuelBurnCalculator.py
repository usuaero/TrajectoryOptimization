import numpy as np
import json
import math as m

def stdatm_si(H):
   levels = np.array([0.,11000.,20000.,32000.,47000.,52000.,61000.,79000.,90000])
   temps = np.array([288.15,216.65,216.65,228.65,270.65,270.65,252.65,180.65,180.65])
   tprime = np.array([-6.5, 0.0, 1.0, 2.8, 0.0, -2.0, -4.0, 0.0])
   tprime = tprime/1000.0
   go = 9.806645
   R = 287.0528
   RE = 6356766.
   po = 101325.
   gamma = 1.4
   
   Z=RE*H/(RE+H)

   Z = max(Z,0.0)
   # print(Z)
   if(Z > 90000):
      T = 180.650
      p = 0.0
   else:
      i = 0
      while (Z >= levels[i]):
         if(tprime[i] == 0):
            if(Z < levels[i+1]):
               T = temps[i]
               p = po*m.exp(-go*(Z-levels[i])/R/temps[i])
            else:
               po = po*m.exp(-go*(levels[i+1]-levels[i])/R/temps[i])
         else:
            if(Z < levels[i+1]):
               T = temps[i] + tprime[i]*(Z-levels[i])
               p = po*((temps[i] + tprime[i]*(Z-levels[i]))/temps[i])**(-go/R/tprime[i])
            else:
               po = po*((temps[i] + tprime[i]*(levels[i+1]-levels[i]))/temps[i])**(-go/R/tprime[i])
         i = i + 1
   rho = p/R/T
   a = m.sqrt(gamma*R*T)
   return Z,T,p,rho,a

def stdatm_english(H):
   H *= 0.3048
   Z,T,p,rho,a = stdatm_si(H)
   Z/=0.3048
   T*=1.8
   p*=0.020885434304801722
   rho*=0.00194032032363104
   a/=0.3048
   return Z,T,p,rho,a


pi = m.pi
gravity = 32.174
print("Fuel Burn Calculator written by D. Hunsaker")
print()
#fn = input("Enter Input Filename: ")
fn = "input.json"

print("Reading aircraft information...")
json_string=open(fn).read()
json_vals = json.loads(json_string)

#Settings
nsteps = json_vals["settings"]["discretization_steps"]
trajfile = json_vals["settings"]["trajectory_input_file"]
trajinfo = np.loadtxt(trajfile,skiprows=1)
initial_altitude = trajinfo[0,1]
initial_velocity = trajinfo[0,2]
total_distance = np.max(trajinfo[:,0])
print("  initial altitude [ft] = ",initial_altitude)
print("initial velocity [ft/s] = ",initial_velocity)
print("    total distance [ft] = ",total_distance)
dx = total_distance/(nsteps-1)
print("         step size [ft] = ",dx)

#Aircraft Information
SW = json_vals["aircraft"]["wing_area[ft^2]"]
b = json_vals["aircraft"]["wing_span[ft]"]
RA = b**2/SW
final_weight = json_vals["aircraft"]["final_weight[lbf]"]

#Aerodynamics
CD0 = json_vals["aerodynamics"]["CD0"]
CD1 = json_vals["aerodynamics"]["CD1"]
CD2 = json_vals["aerodynamics"]["CD2"]
CM1 = json_vals["aerodynamics"]["CM1"]
CM2 = json_vals["aerodynamics"]["CM2"]

#Propulsion
ne = json_vals["propulsion"]["engine_count"]
q = json_vals["propulsion"]["q"]
CT = json_vals["propulsion"]["CT[slug/lbf-s]"]
N1 = json_vals["propulsion"]["N1"]
T0 = json_vals["propulsion"]["T0[lbf]"]
mm = json_vals["propulsion"]["m"]
a1 = json_vals["propulsion"]["a1"]
a2 = json_vals["propulsion"]["a2"]

dummy1, temperature_0, dummy3, dummy4, dummy5 = stdatm_english(0.0)

x = total_distance
weight = final_weight
altitude_prev = np.interp(x,trajinfo[:,0],trajinfo[:,1])
total_time = 0
with open("output.txt", 'w') as output_file:
    output_file.write('x[ft]             altitude[ft]             V[ft/s]             Density[slug/ft^3]            Temp[R]            Mach            Tmax[lbf]               qT               dH/dx           Gamma[deg]             Weight[lbf]                  CL        CD       Drag[lbf]           Thrust[lbf]')

    for i in range(nsteps):
        altitude = np.interp(x,trajinfo[:,0],trajinfo[:,1])
        velocity = np.interp(x,trajinfo[:,0],trajinfo[:,2])
        dummy1, temperature, dummy3, rho, sound = stdatm_english(altitude)
        Mach = velocity/sound
        Tmax = ne*N1*T0*(temperature/temperature_0)**mm*(1.0 + a1*velocity + a2*velocity**2)
        qT = CT*(temperature/temperature_0)**(0.5)*Mach**q

        altitude_next = np.interp(x-dx,trajinfo[:,0],trajinfo[:,1])
        dhdx = (altitude_prev - altitude_next)/dx/2.0
        if(i == 0):
            dhdx = (altitude - altitude_next)/dx
        if(i == nsteps-1):
            dhdx = (altitude_prev - altitude)/dx

        gamma = m.atan(dhdx)
        CL = weight*m.cos(gamma)/0.5/rho/velocity**2/SW
        CD = (CD0 + CD1*CL + CD2*CL**2)*(1.0 + CM1*Mach**CM2)
        Drag = CD*0.5*rho*velocity**2*SW
        Thrust = Drag + weight*m.sin(gamma)
        if(Thrust > Tmax):
            print("Error: Required Thrust > Tmax by this ammount: ",Tmax-Thrust)
        xdot = velocity*m.cos(gamma)
        hdot = velocity*m.sin(gamma)
        Wdot = min(0.0,-qT*Thrust*gravity) #min checks to make sure we don't create fuel
        dWdx = -Wdot/xdot
        dt = dx/xdot

        output_file.write("\n")
        output_file.write("{:>20.12E}".format(x))
        output_file.write("{:>20.12E}".format(altitude))
        output_file.write("{:>20.12E}".format(velocity))
        output_file.write("{:>20.12E}".format(rho))
        output_file.write("{:>20.12E}".format(temperature))
        output_file.write("{:>20.12E}".format(Mach))
        output_file.write("{:>20.12E}".format(Tmax))
        output_file.write("{:>20.12E}".format(qT))
        output_file.write("{:>20.12E}".format(dhdx))
        output_file.write("{:>20.12E}".format(gamma*180.0/pi))
        output_file.write("{:>20.12E}".format(weight))
        output_file.write("{:>20.12E}".format(CL))
        output_file.write("{:>20.12E}".format(CD))
        output_file.write("{:>20.12E}".format(Drag))
        output_file.write("{:>20.12E}".format(Thrust))

        x -= dx
        total_time += dt
        if(i<nsteps-1):
            weight += dWdx*dx
        altitude_prev = altitude

total_fuel = weight - final_weight
print("Total Fuel [lbf] = ",total_fuel)
print("Total Time [hr] = ",total_time/60.0/60.0)

