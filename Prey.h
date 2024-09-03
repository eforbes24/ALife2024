#pragma once

#include "CTRNN.h"

// The Prey class declaration

class Prey {
	public:
		// The constructor
		Prey(int networksize, double gain, double s_width, double frate, double feff, double metaloss, double birth_thresh)
		{
			Set(networksize, gain, s_width, frate, feff, metaloss, birth_thresh);
		};
		Prey() = default;
		// The destructor
		~Prey() {};

		// Accessors
		double Position(void) {return pos;};
		void SetPosition(double newpos) {pos = newpos;};
		void SetSensorWeight(int to, double value) {sensorweights[to] = value;};
		void SetSensorState(double fstate, double pstate) {f_sensor = fstate;
			p_sensor = pstate;};

		// Control
        void Set(int networksize, double gain, double s_width, double frate, double feff, double metaloss, double birth_thresh);
		void Reset(double initpos, double initstate);
		void Sense(TVector<double> &food_pos, TVector<double> &pred_loc);
		void Step(double StepSize, TVector<double> &WorldFood);

		int size;
		double pos, gain, f_sensor, p_sensor, s_width, pastpos, state, frate, feff, metaloss, birth_thresh, s_scalar, a_scalar,
		munchrate, birthrate, snackflag;
		bool death, birth;
		TVector<double> sensorweights;
		CTRNN NervousSystem;
};
