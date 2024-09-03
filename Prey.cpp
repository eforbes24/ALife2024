// Eden Forbes
// MinCogEco Prey

#include "Prey.h"
#include "Predator.h"
#include "random.h"
#include "CTRNN.h"

// Constants
const double SpaceSize = 5000;
const double HalfSpace = SpaceSize/2;

// *******
// Control
// *******

// Init the agent
void Prey::Set(int networksize, double prey_gain, double prey_s_width, double prey_frate, double prey_feff, double prey_metaloss, double prey_b_thresh)
{
	size = networksize;
	gain = prey_gain; 
	sensorweights.SetBounds(1, 3*size);
	sensorweights.FillContents(0.0);
	pos = 0.0;
	pastpos = 0.0;
	f_sensor = 0.0;
    p_sensor = 0.0;
    s_width = prey_s_width;
    state = 1.0;
    frate = prey_frate;
    feff = prey_feff;
    metaloss = prey_metaloss;
    death = false;
    birth = false;
    birth_thresh = prey_b_thresh;
    s_scalar = 0.0;
    a_scalar = 0.0;
    // interaction rates
    munchrate = 0.0;
    birthrate = 0.0;
    snackflag = 0.0;
}

// Reset the state of the agent
void Prey::Reset(double initpos, double initstate)
{
	pos = initpos;
	pastpos = initpos;
	f_sensor = 0.0;
    p_sensor = 0.0;
    state = initstate;
    death = false;
    birth = false;
	NervousSystem.RandomizeCircuitState(0.0,0.0);
}

// Sense 
void Prey::Sense(TVector<double> &food_pos, TVector<double> &pred_loc)
{
	// Sense
	double mindistL = 99999;
    double mindistR = 99999;

    for (int i = 0; i < food_pos.Size(); i++){
        double d = food_pos[i] - pos;
        if (d < 0 && d >= -HalfSpace){
            // Closest to the left side, distance is as calculated
            if (abs(d) < mindistL){
                mindistL = abs(d);
            }
        }
        else if (d < 0 && d < -HalfSpace){
            // Closest to the right side, distance is total area + - left side distance 
            d = SpaceSize + d;
            if (d < mindistR){
                mindistR = d;
            }
        }
        else if (d > 0 && d > HalfSpace){
            // Closest to the left side, distance is -total area + right side distance
            d = -SpaceSize + d;
            if (abs(d) < mindistL){
                mindistL = abs(d);
            }
        }
        else if (d > 0 && d <= HalfSpace){
            // Closest to the right side, distance is as calculated
            if (d < mindistR){
                mindistR = d;
            }
        }
        else if (d == 0){
            d = 0;
        }
        else{
            printf("Food pos size = %d\n", food_pos.Size());
            printf("Error in prey food sensing\n");
            printf("d = %f\n", d);
        }
    }
    // Cumulate, distance fits in the Gaussian as the difference of the mean (position) and the state (food position)
    // Negate left so negative sensor reading says food left, positive says food right, zero says no food or food on both sides.
	f_sensor = -2*exp(-(mindistL) * (mindistL) / (2 * s_width * s_width)) + 2*exp(-(mindistR) * (mindistR) / (2 * s_width * s_width));
    // printf("Prey food sensor = %f\n", f_sensor);

    // Sense Predator
	double PmindistL = 99999;
    double PmindistR = 99999;
    for (int i = 0; i < pred_loc.Size(); i++){
        double d = pred_loc[i] - pos;
        if (d < 0 && d >= -HalfSpace){
            // Closest to the left side, distance is as calculated
            if (abs(d) < PmindistL){
                PmindistL = abs(d);
            }
        }
        else if (d < 0 && d < -HalfSpace){
            // Closest to the right side, distance is total area + - left side distance 
            d = SpaceSize + d;
            if (d < PmindistR){
                PmindistR = d;
            }
        }
        else if (d > 0 && d > HalfSpace){ 
            // Closest to the left side, distance is -total area + right side distance
            d = -SpaceSize + d;
            if (abs(d) < PmindistL){
                PmindistL = abs(d);
            }
        }
        else if (d > 0 && d <= HalfSpace){
            // Closest to the right side, distance is as calculated
            if (d < PmindistR){
                PmindistR = d;
            }
        }
        else if (d == 0){
            d = 0;
        }
        else{
            printf("pred pos size = %d\n", pred_loc.Size());
            printf("Error in prey predator sensing\n");
            printf("d = %f\n", d);
        }
    }
    p_sensor = -2*exp(-(PmindistL) * (PmindistL) / (2 * s_width * s_width)) + 2*exp(-(PmindistR) * (PmindistR) / (2 * s_width * s_width));
    // printf("Prey food sensor = %f\n", f_sensor);
    // printf("Prey pred sensor = %f\n", p_sensor);
}

// Step
void Prey::Step(double StepSize, TVector<double> &WorldFood)
{
    // Remember past position
    pastpos = pos;
    double N1IP = f_sensor*sensorweights[1] + p_sensor*sensorweights[2] + state*sensorweights[3];
    double N2IP = f_sensor*sensorweights[4] + p_sensor*sensorweights[5] + state*sensorweights[6];
    double N3IP = f_sensor*sensorweights[7] + p_sensor*sensorweights[8] + state*sensorweights[9];
    // Give each interneuron its sensory input
    NervousSystem.SetNeuronExternalInput(1, N1IP);
    NervousSystem.SetNeuronExternalInput(2, N2IP);
    NervousSystem.SetNeuronExternalInput(3, N3IP);
	// Update the nervous system
	NervousSystem.EulerStep(StepSize);
	pos += StepSize * gain * (NervousSystem.NeuronOutput(2) - NervousSystem.NeuronOutput(1));

// Update State if the agent passed food
    if (pastpos < pos){
        if (pos > WorldFood.Size()){
            pos = pos - WorldFood.Size();
            int f = floor(pos);
            int c = ceil(pastpos);
            for (int i = c; i < WorldFood.Size(); i++){
                if (WorldFood[i] > 0){
                    state += feff;
                    WorldFood[i] -= frate;
                    snackflag += 1;
                    }
                }
            for (int i = 1; i <= f; i++){
                if (WorldFood[i] > 0){
                    state += feff;
                    WorldFood[i] -= frate;
                    snackflag += 1;
                    }
                }
        }
        else{
            for (int i = 1; i < WorldFood.Size(); i++){
                if (i > pastpos && i <= pos){
                    if (WorldFood[i] > 0){
                        state += feff;
                        WorldFood[i] -= frate;
                        snackflag += 1;
                        }
                    }
                }
        }
    }
    else if (pastpos > pos){
        if (pos < 0){
            pos = pos + WorldFood.Size();
            int f = floor(pastpos);
            int c = ceil(pos);
            for (int i = c; i < WorldFood.Size(); i++){
                if (WorldFood[i] > 0){
                    state += feff;
                    WorldFood[i] -= frate;
                    snackflag += 1;
                    }
                }
            for (int i = 1; i <= f; i++){
                if (WorldFood[i] > 0){
                    state += feff;
                    WorldFood[i] -= frate;
                    snackflag += 1;
                    }
                }
        }
        else{
            for (int i = 1; i < WorldFood.Size(); i++){
                if (i < pastpos && i >= pos){
                    if (WorldFood[i] > 0){
                        state += feff;
                        WorldFood[i] -= frate;
                        snackflag += 1;
                        }
                    }
                }
        }
    }
    // Lose state over time
    // Physiological one needs fixing
    // state -= metaloss * (pow(s_scalar,2)/s_width) * (pow(a_scalar,2)/gain);
    state -= metaloss;

    // Birth & Death
    if (state <= 0){
        death = true;
    }
    if (state > birth_thresh){
        birth = true;
    }
}

