#pragma once

#include "CTRNN.h"
#include "Prey.h"

// The Predator class declaration

class Predator {
	public:
		// The constructor
		
		Predator(int networksize, double gain, double s_width, double frate, double feff, double metaloss, double birth_thresh, double handling_time)
		{
			Set(networksize, gain, s_width, frate, feff, metaloss, birth_thresh, handling_time);
		};
		Predator() = default;
		// The destructor
		~Predator() {};

		// Accessors
		double Position(void) {return pos;};
		void SetPosition(double newpos) {pos = newpos;};
		void SetSensorState(double state) {sensor = state;};

		// Control
        void Set(int networksize, double gain, double s_width, double frate, double feff, double metaloss, double b_thresh, double handling_time);
		void Reset(double initpos, double initstate);
		void Sense(TVector<double> &prey_loc);
		void Step(double StepSize, TVector<double> &WorldFood, TVector<Prey> &preylist);

		int size;
		double pos, gain, sensor, s_width, pastpos,frate, handling_time, 
		handling_counter, munchrate, birthrate, snackflag, birth_thresh, s_scalar, a_scalar, feff, metaloss, state;
		bool handling, birth, death;
		TVector<double> sensorweights;
		CTRNN NervousSystem;
		Prey prey;
};