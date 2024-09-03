// Eden Forbes
// MinCogEco Script

// ***************************************
// INCLUDES
// ***************************************

#include "Prey.h"
#include "Predator.h"
#include "random.h"
#include "TSearch.h"
#include <iostream>
#include <iomanip> 
#include <vector>
#include <string>
#include <list>

// ================================================
// A. PARAMETERS & GEN-PHEN MAPPING
// ================================================

// Run constants
// Make sure SpaceSize is also specified in the Prey.cpp and Predator.cpp files
const int SpaceSize = 5000;
const int finalCC = 2;
const int CC = 2;
const int minCC = 0;
const int maxCC = 5;
// 0-Indexed (0 = 1)
const int start_prey = 0;
const int start_pred = 0;

// Time Constants
// Evolution Step Size:
const double StepSize = 0.1;
// Analysis Step Size:
const double BTStepSize = 0.1;
// Evolution Run Time:
const double RunDuration = 10000;
// Behavioral Trace Run Time:
const double PlotDuration = 3000;
// EcoRate Collection Run Time:
const double RateDuration = 50000;
// Sensory Sample Run Time:
const double SenseDuration = 2000; // 2000

// EA params
const int PREY_POPSIZE = 20;
const int PRED_POPSIZE = 20;
const int GENS = 200;
const double MUTVAR = 0.1;
const double CROSSPROB = 0.0;
const double EXPECTED = 1.1;
const double ELITISM = 0.02;
// Number of trials per trial type (there are maxCC+1 * 2 trial types)
const double n_trials = 10.0;

// Nervous system params
const int prey_netsize = 3; 
const int pred_netsize = 3; 
// weight range 
const double WR = 16.0;
// sensor range
const double SR = 20.0;
// bias range
const double BR = 16.0;
// time constant min max
const double TMIN = 0.5;
const double TMAX = 20.0;
// Sensory Width Scalar Range
const double WMIN = 0.5;
const double WMAX = 2.0;
// Gain Scalar Range
const double GMIN = 0.5;
const double GMAX = 2.0;
// Prey (Weights + TCs & Biases + SensorWeights + PhysiologicalParams) + Pred (Weights + TCs & Biases + SensorWeights + PhysiologicalParams)
const int PreyVectSize = (prey_netsize*prey_netsize + 2*prey_netsize + 3*prey_netsize + 2);
const int PredVectSize = (pred_netsize*pred_netsize + 2*pred_netsize + 2*pred_netsize + 2);

// Producer Parameters
const double G_Rate = 0.001*StepSize;
const double BT_G_Rate = 0.001*BTStepSize;

// Prey Sensory Parameters
const double prey_gain = 3.0;
const double prey_s_width = 100.0;

// Prey Metabolic Parameters
const double prey_loss_scalar = 5.0;
const double prey_frate = 0.15;
const double prey_feff = 0.1;
const double prey_repo = 1.5;
const double prey_b_thresh = 3.0;
const double prey_metaloss = ((prey_feff*(CC+1))/(SpaceSize/(prey_gain*StepSize))) * prey_loss_scalar;
const double prey_BT_metaloss = ((prey_feff*(CC+1))/(SpaceSize/(prey_gain*BTStepSize))) * prey_loss_scalar;

// Predator Sensory Parameters 
const double pred_gain = 3.0;
const double pred_s_width = 100.0;

// Predator Metabolic Parameters
const double pred_loss_scalar = 1.0;
const double pred_frate = 1.0;
const double pred_feff = 0.8;
const double pred_repo = 2.5;
const double pred_b_thresh = 5.0;
const double pred_metaloss = ((pred_feff*(CC+1))/(SpaceSize/(pred_gain*StepSize))) * pred_loss_scalar;
const double pred_BT_metaloss = ((pred_feff*(CC+1))/(SpaceSize/(pred_gain*BTStepSize))) * pred_loss_scalar;
const double pred_handling_time = 10.0/StepSize;
const double pred_BT_handling_time = 10.0/BTStepSize;

// ------------------------------------
// Genotype-Phenotype Mapping Function
// ------------------------------------
void GenPhenMapping(TVector<double> &gen, TVector<double> &phen, double species)
{
    double size = 0;
    if (gen.Size() == PreyVectSize){
        size = prey_netsize;
    }
    else{
        size = pred_netsize;
    }
	int k = 1;
	// Time-constants
	for (int i = 1; i <= size; i++) {
		phen(k) = MapSearchParameter(gen(k), TMIN, TMAX);
		k++;
	}
	// Bias
	for (int i = 1; i <= size; i++) {
		phen(k) = MapSearchParameter(gen(k), -BR, BR);
		k++;
	}
	// Weights
	for (int i = 1; i <= size; i++) {
		for (int j = 1; j <= size; j++) {
            phen(k) = MapSearchParameter(gen(k), -WR, WR);
			k++;
		}
	}
	// Sensor Weights
    if(species == 0){
        for (int i = 1; i <= size*3; i++) {
            phen(k) = MapSearchParameter(gen(k), -SR, SR);
            k++;
        }
    }
    else{
        for (int i = 1; i <= size*2; i++) {
            phen(k) = MapSearchParameter(gen(k), -SR, SR);
            k++;
        }
    }
    phen(k) = MapSearchParameter(gen(k), WMIN, WMAX);
    k++;
    phen(k) = MapSearchParameter(gen(k), GMIN, GMAX);
}
// ================================================
// B. TASK ENVIRONMENT & FITNESS FUNCTION
// ================================================
double PreyTest(TVector<double> &genotype, RandomState &rs, TVector<double> &bestpred) 
{
    // condition 0 = evaluate prey; condition 1 = evaluate predator
    // Set running outcome variable
    // For Minimum Fitness
    // double outcome = 99999999999.0;
    // For Average Fitness
    double outcome = 0.0;
    // Translate genotype to phenotype
	TVector<double> preyphenotype;
	preyphenotype.SetBounds(1, PreyVectSize);
	GenPhenMapping(genotype, preyphenotype, 0);
    TVector<double> predphenotype;
	predphenotype.SetBounds(1, PredVectSize);
    GenPhenMapping(bestpred, predphenotype, 1);
    // Initialize Prey & Predator agents
    Prey Agent1(prey_netsize, prey_gain, prey_s_width, prey_frate, prey_feff, prey_metaloss, prey_b_thresh);
    Predator Agent2(pred_netsize, pred_gain, pred_s_width, pred_frate, pred_feff, pred_metaloss, pred_b_thresh, pred_handling_time);
    // Set Prey nervous system
    Agent1.NervousSystem.SetCircuitSize(prey_netsize);
    Agent2.NervousSystem.SetCircuitSize(pred_netsize);
    int k = 1;
    // Prey Time-constants
    for (int i = 1; i <= prey_netsize; i++) {
        Agent1.NervousSystem.SetNeuronTimeConstant(i,preyphenotype(k));
        k++;
    }
    // Prey Biases
    for (int i = 1; i <= prey_netsize; i++) {
        Agent1.NervousSystem.SetNeuronBias(i,preyphenotype(k));
        k++;
    }
    // Prey Neural Weights
    for (int i = 1; i <= prey_netsize; i++) {
        for (int j = 1; j <= prey_netsize; j++) {
            Agent1.NervousSystem.SetConnectionWeight(i,j,preyphenotype(k));
            k++;
        }
    }
    // Prey Sensor Weights
    for (int i = 1; i <= prey_netsize*3; i++) {
        Agent1.sensorweights[i] = preyphenotype(k);
        k++;
    }
    Agent1.s_scalar = preyphenotype(k);
    k++;
    Agent1.a_scalar = preyphenotype(k);
    k++;
    k = 1;
    // Prey Time-constants
    for (int i = 1; i <= pred_netsize; i++) {
        Agent2.NervousSystem.SetNeuronTimeConstant(i,predphenotype(k));
        k++;
    }
    // Prey Biases
    for (int i = 1; i <= pred_netsize; i++) {
        Agent2.NervousSystem.SetNeuronBias(i,predphenotype(k));
        k++;
    }
    // Prey Neural Weights
    for (int i = 1; i <= pred_netsize; i++) {
        for (int j = 1; j <= pred_netsize; j++) {
            Agent2.NervousSystem.SetConnectionWeight(i,j,predphenotype(k));
            k++;
        }
    }
    // Prey Sensor Weights
    for (int i = 1; i <= pred_netsize*2; i++) {
        Agent2.sensorweights[i] = predphenotype(k);
        k++;
    }
    Agent2.s_scalar = predphenotype(k);
    k++;
    Agent2.a_scalar = predphenotype(k);
    k++;
    // Set Trial Structure - Fixed CC
    double CoexistTrials = n_trials;
    // Run Simulation
    for (int trial = 0; trial < CoexistTrials; trial++){
        // Reset Prey agent, randomize its location
        Agent1.Reset(rs.UniformRandomInteger(0,SpaceSize), 3.0);
        Agent2.Reset(rs.UniformRandomInteger(0,SpaceSize), 5.0);
        // Seed preylist with starting population
        TVector<Prey> preylist(0,0);
        preylist[0] = Agent1;
        // for (int i = 0; i < start_prey; i++){
        //     Prey newprey(prey_netsize, prey_gain, prey_s_width, prey_frate, prey_feff, prey_metaloss, prey_b_thresh);
        //     // Reset Prey agent, randomize its location
        //     newprey.Reset(rs.UniformRandomInteger(0,SpaceSize), 3.0);
        //     // Copy over nervous system
        //     newprey.NervousSystem = Agent1.NervousSystem;
        //     newprey.sensorweights = Agent1.sensorweights;
        //     // Add to preylist
        //     preylist.SetBounds(0, preylist.Size());
        //     preylist[preylist.Size()-1] = newprey;
        // }
        // Seed predlist with starting population
        TVector<Predator> predlist(0,0);
        predlist[0] = Agent2;
        // for (int i = 0; i < start_pred; i++){
        //     Predator newpred(pred_netsize, pred_gain, pred_s_width, pred_frate, pred_feff, pred_metaloss, pred_b_thresh, pred_handling_time);
        //     // Reset Predator agent, randomize its location
        //     newpred.Reset(rs.UniformRandomInteger(0,SpaceSize), 5.0);
        //     // Copy over nervous system
        //     newpred.NervousSystem = Agent2.NervousSystem;
        //     newpred.sensorweights = Agent2.sensorweights;
        //     // Add to predlist
        //     predlist.SetBounds(0, predlist.Size());
        //     predlist[predlist.Size()-1] = newpred;
        // }
        // Initialize Producers, fill world to carrying capacity
        TVector<double> food_pos;
        TVector<double> WorldFood(1, SpaceSize);
        WorldFood.FillContents(0.0);
        for (int i = 0; i <= CC; i++){
            int f = rs.UniformRandomInteger(1,SpaceSize);
            WorldFood[f] = 1.0;
            food_pos.SetBounds(0, food_pos.Size());
            food_pos[food_pos.Size()-1] = f;
        }
        // Set Clocks & trial outcome variables
        double clock = 0.0;
        double prey_snacks = 0.0;
        double pred_snacks = 0.0;
        double prey_outcome = RunDuration;
        double pred_outcome = RunDuration;
        double final_state = 0.0;
        double prey_birth_count = 0.0;
        double pred_birth_count = 0.0;
        // Run a Trial
        for (double time = 0; time < RunDuration; time += StepSize){
            // Remove any consumed food from food list
            TVector<double> dead_food(0,-1);
            for (int i = 0; i < food_pos.Size(); i++){
                if (WorldFood[food_pos[i]] <= 0){
                    dead_food.SetBounds(0, dead_food.Size());
                    dead_food[dead_food.Size()-1] = food_pos[i];
                }
            }
            if (dead_food.Size() > 0){
                for (int i = 0; i < dead_food.Size(); i++){
                    food_pos.RemoveFood(dead_food[i]);
                    food_pos.SetBounds(0, food_pos.Size()-2);
                }
            }
            // Chance for new food to grow
            // Carrying capacity is 0 indexed, add 1 for true amount
            for (int i = 0; i < CC+1 - food_pos.Size(); i++){
                double c = rs.UniformRandom(0,1);
                if (c <= G_Rate){
                    int f = rs.UniformRandomInteger(1,SpaceSize);
                    WorldFood[f] = 1.0;
                    food_pos.SetBounds(0, food_pos.Size());
                    food_pos[food_pos.Size()-1] = f;
                }
            }
            // Update Prey Positions
            TVector<double> prey_pos;
            for (int i = 0; i < preylist.Size(); i++){
                prey_pos.SetBounds(0, prey_pos.Size());
                prey_pos[prey_pos.Size()-1] = preylist[i].pos;
            }
            // Predator Sense & Step
            for (int i = 0; i < predlist.Size(); i++){
                // printf("Predator metaloss check: %f\n", predlist[i].metaloss);
                // printf("Predator fill before: %f\n", predlist[i].state);
                predlist[i].Sense(prey_pos);
                predlist[i].Step(StepSize, WorldFood, preylist);
                // printf("Predator fill after: %f\n", predlist[i].state);
                if (predlist[i].snackflag > 0){
                    pred_snacks += predlist[i].snackflag;
                    predlist[i].snackflag = 0;
                }
                if (predlist[i].birth == true){
                    predlist[i].state = predlist[i].state - prey_repo;
                    pred_birth_count += 1;
                    predlist[i].birth = false;
                //     // ONLY WITH POP
                //     Predator newpred(pred_netsize, pred_gain, pred_s_width, pred_frate, pred_feff, pred_metaloss, pred_b_thresh, pred_handling_time);
                //     newpred.NervousSystem = predlist[i].NervousSystem;
                //     newpred.sensorweights = predlist[i].sensorweights;
                //     newpred.Reset(predlist[i].pos+2, pred_repo);
                //     newpredlist.SetBounds(0, newpredlist.Size());
                //     newpredlist[newpredlist.Size()-1] = newpred;
                }
                if (predlist[i].death == true){
                    predlist[i].Reset(rs.UniformRandomInteger(0,SpaceSize), 5.0);
                    predlist[i].death = false;
                }
            }
            
            // Update Predator Positions
            TVector<double> pred_pos;
            for (int i = 0; i < predlist.Size(); i++){
                pred_pos.SetBounds(0, pred_pos.Size());
                pred_pos[pred_pos.Size()-1] = predlist[i].pos;
            }
            // Prey Sense & Step
            TVector<int> preydeaths;
            for (int i = 0; i < preylist.Size(); i++){
                // printf("Prey metaloss check: %f\n", preylist[i].metaloss);
                // printf("Prey fill before: %f\n", preylist[i].state);
                preylist[i].Sense(food_pos, pred_pos);
                preylist[i].Step(StepSize, WorldFood);
                // printf("Prey fill after: %f\n", preylist[i].state);
                if (preylist[i].snackflag > 0){
                    prey_snacks += preylist[i].snackflag;
                    preylist[i].snackflag = 0;
                }
                if (preylist[i].birth == true){
                    preylist[i].state = preylist[i].state - prey_repo;
                    prey_birth_count += 1;
                    preylist[i].birth = false;
                //     // ONLY WITH POP
                //     Prey newprey(prey_netsize, prey_gain, prey_s_width, prey_frate, prey_feff, prey_metaloss, prey_b_thresh);
                //     newprey.NervousSystem = preylist[i].NervousSystem;
                //     newprey.sensorweights = preylist[i].sensorweights;
                //     newprey.Reset(preylist[i].pos+2, prey_repo);
                //     newpreylist.SetBounds(0, newpreylist.Size());
                //     newpreylist[newpreylist.Size()-1] = newprey;
                }
                if (preylist[i].death == true){
                    preydeaths.SetBounds(0, preydeaths.Size());
                    preydeaths[preydeaths.Size()-1] = i;
                }
            }
            // Update clocks
            clock += StepSize;
            // Update prey list with new prey list and deaths
            if (preydeaths.Size() > 0){
                for (int i = 0; i <= preydeaths.Size()-1; i++){
                    preylist.RemoveItem(preydeaths[i]);
                    preylist.SetBounds(0, preylist.Size()-2);
                }
            }
            // ONLY WITH POPS
            // if (newpredlist.Size() > 0){
            //     for (int i = 0; i <= newpredlist.Size()-1; i++){
            //         predlist.SetBounds(0, predlist.Size());
            //         predlist[predlist.Size()-1] = newpredlist[i];
            //     }
            // }
            // Check for prey population collapse
            if (preylist.Size() <= 0){
                final_state = 0.0;
                prey_outcome = clock;
                break;
            }
            // Reset lists for next step
            else{
                final_state = preylist[0].state;
                preydeaths.~TVector();
                prey_pos.~TVector();
                pred_pos.~TVector();
                dead_food.~TVector();
            }
        }
        double runmeasure = (clock/RunDuration);
        // FOR POP
        // double popmeasure = (running_pop/(RunDuration/StepSize))/100;
        // FOR IND
        double popmeasure = (prey_birth_count+final_state)/100;
        double fitmeasure = runmeasure + popmeasure;
        // // Keep minimum fitness value across trials
        // if (fitmeasure < outcome){
        //     outcome = fitmeasure;
        // }
        // // Take average fitness value across trials
        outcome += fitmeasure;
    }
    double final_outcome = outcome/n_trials;
    return final_outcome;
}

double PredTest(TVector<double> &genotype, RandomState &rs, TVector<double> &bestprey) 
{
    // condition 0 = evaluate prey; condition 1 = evaluate predator
    // Set running outcome variable
    // For Minimum Fitness
    // double outcome = 99999999999.0;
    // For Average Fitness
    double outcome = 0.0;
    // Translate genotype to phenotype
	TVector<double> preyphenotype;
	preyphenotype.SetBounds(1, PreyVectSize);
	GenPhenMapping(bestprey, preyphenotype, 0);
    TVector<double> predphenotype;
	predphenotype.SetBounds(1, PredVectSize);
    GenPhenMapping(genotype, predphenotype, 1);
    // Initialize Prey & Predator agents
    Prey Agent1(prey_netsize, prey_gain, prey_s_width, prey_frate, prey_feff, prey_metaloss, prey_b_thresh);
    Predator Agent2(pred_netsize, pred_gain, pred_s_width, pred_frate, pred_feff, pred_metaloss, pred_b_thresh, pred_handling_time);
    // Set Prey nervous system
    Agent1.NervousSystem.SetCircuitSize(prey_netsize);
    Agent2.NervousSystem.SetCircuitSize(pred_netsize);
    int k = 1;
    // Prey Time-constants
    for (int i = 1; i <= prey_netsize; i++) {
        Agent1.NervousSystem.SetNeuronTimeConstant(i,preyphenotype(k));
        k++;
    }
    // Prey Biases
    for (int i = 1; i <= prey_netsize; i++) {
        Agent1.NervousSystem.SetNeuronBias(i,preyphenotype(k));
        k++;
    }
    // Prey Neural Weights
    for (int i = 1; i <= prey_netsize; i++) {
        for (int j = 1; j <= prey_netsize; j++) {
            Agent1.NervousSystem.SetConnectionWeight(i,j,preyphenotype(k));
            k++;
        }
    }
    // Prey Sensor Weights
    for (int i = 1; i <= prey_netsize*3; i++) {
        Agent1.sensorweights[i] = preyphenotype(k);
        k++;
    }
    Agent1.s_scalar = preyphenotype(k);
    k++;
    Agent1.a_scalar = preyphenotype(k);
    k++;
    k = 1;
    // Prey Time-constants
    for (int i = 1; i <= pred_netsize; i++) {
        Agent2.NervousSystem.SetNeuronTimeConstant(i,predphenotype(k));
        k++;
    }
    // Prey Biases
    for (int i = 1; i <= pred_netsize; i++) {
        Agent2.NervousSystem.SetNeuronBias(i,predphenotype(k));
        k++;
    }
    // Prey Neural Weights
    for (int i = 1; i <= pred_netsize; i++) {
        for (int j = 1; j <= pred_netsize; j++) {
            Agent2.NervousSystem.SetConnectionWeight(i,j,predphenotype(k));
            k++;
        }
    }
    // Prey Sensor Weights
    for (int i = 1; i <= pred_netsize*2; i++) {
        Agent2.sensorweights[i] = predphenotype(k);
        k++;
    }
    Agent2.s_scalar = predphenotype(k);
    k++;
    Agent2.a_scalar = predphenotype(k);
    k++;
    // Set Trial Structure - Fixed CC
    double CoexistTrials = n_trials;
    // Run Simulation
    for (int trial = 0; trial < CoexistTrials; trial++){
        // Reset Prey agent, randomize its location
        Agent1.Reset(rs.UniformRandomInteger(0,SpaceSize), 3.0);
        Agent2.Reset(rs.UniformRandomInteger(0,SpaceSize), 5.0);
        // Seed preylist with starting population
        TVector<Prey> preylist(0,0);
        preylist[0] = Agent1;
        // for (int i = 0; i < start_prey; i++){
        //     Prey newprey(prey_netsize, prey_gain, prey_s_width, prey_frate, prey_feff, prey_metaloss, prey_b_thresh);
        //     // Reset Prey agent, randomize its location
        //     newprey.Reset(rs.UniformRandomInteger(0,SpaceSize), 3.0);
        //     // Copy over nervous system
        //     newprey.NervousSystem = Agent1.NervousSystem;
        //     newprey.sensorweights = Agent1.sensorweights;
        //     // Add to preylist
        //     preylist.SetBounds(0, preylist.Size());
        //     preylist[preylist.Size()-1] = newprey;
        // }
        // Seed predlist with starting population
        TVector<Predator> predlist(0,0);
        predlist[0] = Agent2;
        // for (int i = 0; i < start_pred; i++){
        //     Predator newpred(pred_netsize, pred_gain, pred_s_width, pred_frate, pred_feff, pred_metaloss, pred_b_thresh, pred_handling_time);
        //     // Reset Predator agent, randomize its location
        //     newpred.Reset(rs.UniformRandomInteger(0,SpaceSize), 5.0);
        //     // Copy over nervous system
        //     newpred.NervousSystem = Agent2.NervousSystem;
        //     newpred.sensorweights = Agent2.sensorweights;
        //     // Add to predlist
        //     predlist.SetBounds(0, predlist.Size());
        //     predlist[predlist.Size()-1] = newpred;
        // }
        // Initialize Producers, fill world to carrying capacity
        TVector<double> food_pos;
        TVector<double> WorldFood(1, SpaceSize);
        WorldFood.FillContents(0.0);
        for (int i = 0; i <= CC; i++){
            int f = rs.UniformRandomInteger(1,SpaceSize);
            WorldFood[f] = 1.0;
            food_pos.SetBounds(0, food_pos.Size());
            food_pos[food_pos.Size()-1] = f;
        }
        // Set Clocks & trial outcome variables
        double clock = 0.0;
        double prey_snacks = 0.0;
        double pred_snacks = 0.0;
        double prey_outcome = RunDuration;
        double pred_outcome = RunDuration;
        double final_state = 0.0;
        double final_dist = 0.0;
        double prey_birth_count = 0.0;
        double pred_birth_count = 0.0;
        // Run a Trial
        for (double time = 0; time < RunDuration; time += StepSize){
            // Remove any consumed food from food list
            TVector<double> dead_food(0,-1);
            for (int i = 0; i < food_pos.Size(); i++){
                if (WorldFood[food_pos[i]] <= 0){
                    dead_food.SetBounds(0, dead_food.Size());
                    dead_food[dead_food.Size()-1] = food_pos[i];
                }
            }
            if (dead_food.Size() > 0){
                for (int i = 0; i < dead_food.Size(); i++){
                    food_pos.RemoveFood(dead_food[i]);
                    food_pos.SetBounds(0, food_pos.Size()-2);
                }
            }
            // Chance for new food to grow
            // Carrying capacity is 0 indexed, add 1 for true amount
            for (int i = 0; i < CC+1 - food_pos.Size(); i++){
                double c = rs.UniformRandom(0,1);
                if (c <= G_Rate){
                    int f = rs.UniformRandomInteger(1,SpaceSize);
                    WorldFood[f] = 1.0;
                    food_pos.SetBounds(0, food_pos.Size());
                    food_pos[food_pos.Size()-1] = f;
                }
            }
            // Update Prey Positions
            TVector<double> prey_pos;
            for (int i = 0; i < preylist.Size(); i++){
                prey_pos.SetBounds(0, prey_pos.Size());
                prey_pos[prey_pos.Size()-1] = preylist[i].pos;
            }
            // Predator Sense & Step
            TVector<int> preddeaths;
            for (int i = 0; i < predlist.Size(); i++){
                predlist[i].Sense(prey_pos);
                predlist[i].Step(StepSize, WorldFood, preylist);
                if (predlist[i].snackflag > 0){
                    pred_snacks += predlist[i].snackflag;
                    predlist[i].snackflag = 0;
                }
                if (predlist[i].birth == true){
                    predlist[i].state = predlist[i].state - prey_repo;
                    pred_birth_count += 1;
                    predlist[i].birth = false;
                //     // ONLY WITH POP
                //     Predator newpred(pred_netsize, pred_gain, pred_s_width, pred_frate, pred_feff, pred_metaloss, pred_b_thresh, pred_handling_time);
                //     newpred.NervousSystem = predlist[i].NervousSystem;
                //     newpred.sensorweights = predlist[i].sensorweights;
                //     newpred.Reset(predlist[i].pos+2, pred_repo);
                //     newpredlist.SetBounds(0, newpredlist.Size());
                //     newpredlist[newpredlist.Size()-1] = newpred;
                }
                if (predlist[i].death == true){
                    preddeaths.SetBounds(0, preddeaths.Size());
                    preddeaths[preddeaths.Size()-1] = i;
                }
            }
            // Update Predator Positions
            TVector<double> pred_pos;
            for (int i = 0; i < predlist.Size(); i++){
                pred_pos.SetBounds(0, pred_pos.Size());
                pred_pos[pred_pos.Size()-1] = predlist[i].pos;
            }
            // Prey Sense & Step
            for (int i = 0; i < preylist.Size(); i++){
                preylist[i].Sense(food_pos, pred_pos);
                preylist[i].Step(StepSize, WorldFood);
                if (preylist[i].snackflag > 0){
                    prey_snacks += preylist[i].snackflag;
                    preylist[i].snackflag = 0;
                }
                if (preylist[i].birth == true){
                    preylist[i].state = preylist[i].state - prey_repo;
                    prey_birth_count += 1;
                    preylist[i].birth = false;
                //     // ONLY WITH POP
                //     Prey newprey(prey_netsize, prey_gain, prey_s_width, prey_frate, prey_feff, prey_metaloss, prey_b_thresh);
                //     newprey.NervousSystem = preylist[i].NervousSystem;
                //     newprey.sensorweights = preylist[i].sensorweights;
                //     newprey.Reset(preylist[i].pos+2, prey_repo);
                //     newpreylist.SetBounds(0, newpreylist.Size());
                //     newpreylist[newpreylist.Size()-1] = newprey;
                }
                if (preylist[i].death == true){
                    Agent1.Reset(rs.UniformRandomInteger(0,SpaceSize), 3.0);
                    preylist[i].death = false;
                }
            }
            // Update clocks
            clock += StepSize;
            // Update prey list with new prey list and deaths
            // ONLY FOR POPS
            // if (newpreylist.Size() > 0){
            //     for (int i = 0; i <= newpreylist.Size()-1; i++){
            //         preylist.SetBounds(0, preylist.Size());
            //         preylist[preylist.Size()-1] = newpreylist[i];
            //     }
            // }
            if (preddeaths.Size() > 0){
                for (int i = 0; i <= preddeaths.Size()-1; i++){
                    predlist.RemoveItem(preddeaths[i]);
                    predlist.SetBounds(0, predlist.Size()-2);
                }
            }
            // ONLY FOR POPS
            // if (newpredlist.Size() > 0){
            //     for (int i = 0; i <= newpredlist.Size()-1; i++){
            //         predlist.SetBounds(0, predlist.Size());
            //         predlist[predlist.Size()-1] = newpredlist[i];
            //     }
            // }
            // Check for predator population collapse
            
            if (predlist.Size() <= 0){
                final_state = 0.0;
                final_dist = abs(predlist[0].pos - preylist[0].pos);
                if (final_dist > SpaceSize/2){
                    final_dist = SpaceSize - final_dist;
                }
                pred_outcome = clock;
                break;
            }
            
            // Reset lists for next step
            else{
                final_state = predlist[0].state;
                final_dist = abs(predlist[0].pos - preylist[0].pos);
                if (final_dist > SpaceSize/2){
                    final_dist = SpaceSize - final_dist;
                }
                preddeaths.~TVector();
                prey_pos.~TVector();
                pred_pos.~TVector();
                dead_food.~TVector();
            }
        }
        double runmeasure = (clock/RunDuration);
        // FOR POP
        // double popmeasure = (running_pop/(RunDuration/StepSize))/100;
        // FOR IND
        double popmeasure = (1/(final_dist+1))/10;
        double fitmeasure = runmeasure + popmeasure;
        // // Keep minimum fitness value across trials
        // if (fitmeasure < outcome){
        //     outcome = fitmeasure;
        // }
        // // Take average fitness value across trials
        outcome += fitmeasure;
    }
    double final_outcome = outcome/n_trials;
    return final_outcome;
}


// ================================================
// C. ADDITIONAL EVOLUTIONARY FUNCTIONS
// ================================================
int IntTerminationFunction(int Generation, double BestPerf, double AvgPerf, double PerfVar)
{
	if (BestPerf >= 0.7) return 1;
	else return 0;
}

int EndTerminationFunction(int Generation, double BestPerf, double AvgPerf, double PerfVar)
{
	if (BestPerf >= 100.0) return 1;
	else return 0;
}

void EvolutionaryRunDisplay(int Generation, double BestPerf, double AvgPerf, double PerfVar)
{
	cout << Generation << " " << BestPerf << " " << AvgPerf << " " << PerfVar << endl;
}

void ResultsDisplay(TSearch &s)
{
	TVector<double> bestVector;
	ofstream BestIndividualFile;

	// Save the genotype of the best individual
	bestVector = s.BestIndividual();
	BestIndividualFile.open("best.gen.dat");
    BestIndividualFile << setprecision(32);
	BestIndividualFile << bestVector << endl;
	BestIndividualFile.close();
}

// ================================================
// D. ANALYSIS FUNCTIONS
// ================================================
// ------------------------------------
// Behavioral Traces
// ------------------------------------

double BehavioralTracesCoexist (TVector<double> &prey_genotype, TVector<double> &pred_genotype, RandomState &rs, int agent) 
{
    // Start output files
    std::string pyfile = "menagerie/TestBatch/analysis_results/ns_";
    std::string pdfile = "menagerie/TestBatch/analysis_results/ns_";
    std::string pypfile = "menagerie/TestBatch/analysis_results/ns_";
    std::string pdpfile = "menagerie/TestBatch/analysis_results/ns_";
    std::string ffile = "menagerie/TestBatch/analysis_results/ns_";
    std::string fpfile = "menagerie/TestBatch/analysis_results/ns_";
    pyfile += std::to_string(agent);
    pdfile += std::to_string(agent);
    pypfile += std::to_string(agent);
    pdpfile += std::to_string(agent);
    ffile += std::to_string(agent);
    fpfile += std::to_string(agent);
    pyfile += "/prey_pos.dat";
    pdfile += "/pred_pos.dat";
    pypfile += "/prey_pop.dat";
    pdpfile += "/pred_pop.dat";
    ffile += "/food_pos.dat";
    fpfile += "/food_pop.dat";
    ofstream preyfile(pyfile);
    ofstream predfile(pdfile);
    ofstream preypopfile(pypfile);
    ofstream predpopfile(pdpfile);
    ofstream foodfile(ffile);
    ofstream foodpopfile(fpfile);
    // Translate to phenotypes
	TVector<double> prey_phenotype;
	TVector<double> pred_phenotype;
	prey_phenotype.SetBounds(1, PreyVectSize);
    pred_phenotype.SetBounds(1, PredVectSize);
	GenPhenMapping(prey_genotype, prey_phenotype, 0);
	GenPhenMapping(pred_genotype, pred_phenotype, 1);
    // Create agents
    Prey Agent1(prey_netsize, prey_gain, prey_s_width, prey_frate, prey_feff, prey_metaloss, prey_b_thresh);
    Predator Agent2(pred_netsize, pred_gain, pred_s_width, pred_frate, pred_feff, pred_metaloss, pred_b_thresh, pred_handling_time);
    // Set Prey nervous system
    Agent1.NervousSystem.SetCircuitSize(prey_netsize);
    Agent2.NervousSystem.SetCircuitSize(pred_netsize);
    int k = 1;
    // Prey Time-constants
    for (int i = 1; i <= prey_netsize; i++) {
        Agent1.NervousSystem.SetNeuronTimeConstant(i,prey_phenotype(k));
        k++;
    }
    // Prey Biases
    for (int i = 1; i <= prey_netsize; i++) {
        Agent1.NervousSystem.SetNeuronBias(i,prey_phenotype(k));
        k++;
    }
    // Prey Neural Weights
    for (int i = 1; i <= prey_netsize; i++) {
        for (int j = 1; j <= prey_netsize; j++) {
            Agent1.NervousSystem.SetConnectionWeight(i,j,prey_phenotype(k));
            k++;
        }
    }
    int j = k;
    // Prey Sensor Weights
    for (int i = 1; i <= prey_netsize*3; i++) {
        Agent1.sensorweights[i] = prey_phenotype(k);
        k++;
    }
    Agent1.s_scalar = prey_phenotype(k);
    k++;
    Agent1.a_scalar = prey_phenotype(k);
    k++;

    k = 1;
    // Predator Time-constants
    for (int i = 1; i <= pred_netsize; i++) {
        Agent2.NervousSystem.SetNeuronTimeConstant(i,pred_phenotype(k));
        k++;
    }
    // Predator Biases
    for (int i = 1; i <= pred_netsize; i++) {
        Agent2.NervousSystem.SetNeuronBias(i,pred_phenotype(k));
        k++;
    }
    // Predator Neural Weights
    for (int i = 1; i <= pred_netsize; i++) {
        for (int j = 1; j <= pred_netsize; j++) {
            Agent2.NervousSystem.SetConnectionWeight(i,j,pred_phenotype(k));
            k++;
        }
    }
    j = k;
    // Predator Sensor Weights
    for (int i = 1; i <= pred_netsize*2; i++) {
        Agent2.sensorweights[i] = pred_phenotype(k);
        k++;
    }
    Agent2.s_scalar = pred_phenotype(k);
    k++;
    Agent2.a_scalar = pred_phenotype(k);
    k++;

    // Run Simulation
    // Reset Agents & Vectors
    Agent1.Reset(rs.UniformRandomInteger(0,SpaceSize), 1.5);
    Agent2.Reset(rs.UniformRandomInteger(0,SpaceSize), 2.5);
    // Seed preylist with starting population
    TVector<Prey> preylist(0,0);
    preylist[0] = Agent1;
    for (int i = 0; i < start_prey; i++){
        Prey newprey(prey_netsize, prey_gain, prey_s_width, prey_frate, prey_feff, prey_BT_metaloss, prey_b_thresh);
        newprey.Reset(rs.UniformRandomInteger(0,SpaceSize), 1.5);
        newprey.NervousSystem = Agent1.NervousSystem;
        newprey.sensorweights = Agent1.sensorweights;
        preylist.SetBounds(0, preylist.Size());
        preylist[preylist.Size()-1] = newprey;
    }
    // Seed predlist with starting population
    TVector<Predator> predlist(0,0);
    predlist[0] = Agent2;
    for (int i = 0; i < start_pred; i++){
        Predator newpred(pred_netsize, pred_gain, pred_s_width, pred_frate, pred_feff, pred_BT_metaloss, pred_b_thresh, pred_BT_handling_time);
        newpred.Reset(rs.UniformRandomInteger(0,SpaceSize), 2.5);
        newpred.NervousSystem = Agent2.NervousSystem;
        newpred.sensorweights = Agent2.sensorweights;
        predlist.SetBounds(0, predlist.Size());
        predlist[predlist.Size()-1] = newpred;
    }
    // Fill World to Carrying Capacity
    TVector<double> food_pos(0,-1);
    TVector<double> WorldFood(1, SpaceSize);
    WorldFood.FillContents(0.0);
    for (int i = 0; i <= CC; i++){
        int f = rs.UniformRandomInteger(1,SpaceSize);
        WorldFood[f] = 1.0;
        food_pos.SetBounds(0, food_pos.Size());
        food_pos[food_pos.Size()-1] = f;
    }
    // Run Simulation
    for (double time = 0; time < PlotDuration; time += BTStepSize){
        // Remove chomped food from food list
        TVector<double> dead_food(0,-1);
        for (int i = 0; i < food_pos.Size(); i++){
            if (WorldFood[food_pos[i]] <= 0){
                dead_food.SetBounds(0, dead_food.Size());
                dead_food[dead_food.Size()-1] = food_pos[i];
            }
        }
        if (dead_food.Size() > 0){
            for (int i = 0; i < dead_food.Size(); i++){
                food_pos.RemoveFood(dead_food[i]);
                food_pos.SetBounds(0, food_pos.Size()-2);
            }
        }
        // Carrying capacity is 0 indexed, add 1 for true amount
        for (int i = 0; i < ((CC+1) - food_pos.Size()); i++){
                double c = rs.UniformRandom(0,1);
                if (c <= BT_G_Rate){
                    int f = rs.UniformRandomInteger(1,SpaceSize);
                    WorldFood[f] = 1.0;
                    food_pos.SetBounds(0, food_pos.Size());
                    food_pos[food_pos.Size()-1] = f;
                }
            }
        // Update Prey Positions
        TVector<double> prey_pos;
        for (int i = 0; i < preylist.Size(); i++){
            prey_pos.SetBounds(0, prey_pos.Size());
            prey_pos[prey_pos.Size()-1] = preylist[i].pos;
        }
        // Predator Sense & Step
        TVector<Predator> newpredlist;
        TVector<int> preddeaths;
        for (int i = 0; i < predlist.Size(); i++){
            predlist[i].Sense(prey_pos);
            predlist[i].Step(BTStepSize, WorldFood, preylist);
            if (predlist[i].birth == true){
                predlist[i].state = predlist[i].state - pred_repo;
                predlist[i].birth = false;
                // FOR POPS ONLY
                Predator newpred(pred_netsize, pred_gain, pred_s_width, pred_frate, pred_feff, pred_BT_metaloss, pred_b_thresh, pred_BT_handling_time);
                newpred.Reset(predlist[i].pos+2, pred_repo);
                newpred.NervousSystem = Agent2.NervousSystem;
                newpred.sensorweights = Agent2.sensorweights;
                newpredlist.SetBounds(0, predlist.Size());
                newpredlist[predlist.Size()-1] = newpred;
            }
            if (predlist[i].death == true){
                preddeaths.SetBounds(0, preddeaths.Size());
                preddeaths[preddeaths.Size()-1] = i;
            }
        }
        // Update Predator Positions
        TVector<double> pred_pos;
        for (int i = 0; i < predlist.Size(); i++){
            pred_pos.SetBounds(0, pred_pos.Size());
            pred_pos[pred_pos.Size()-1] = predlist[i].pos;
        }
        // Prey Sense & Step
        TVector<Prey> newpreylist;
        TVector<int> preydeaths;
        for (int i = 0; i < preylist.Size(); i++){
            preylist[i].Sense(food_pos, pred_pos);
            preylist[i].Step(BTStepSize, WorldFood);
            if (preylist[i].birth == true){
                preylist[i].state = preylist[i].state - prey_repo;
                preylist[i].birth = false;
                // FOR POPS ONLY
                Prey newprey(prey_netsize, prey_gain, prey_s_width, prey_frate, prey_feff, prey_metaloss, prey_b_thresh);
                newprey.NervousSystem = preylist[i].NervousSystem;
                newprey.sensorweights = preylist[i].sensorweights;
                newprey.Reset(preylist[i].pos+2, prey_repo);
                newpreylist.SetBounds(0, newpreylist.Size());
                newpreylist[newpreylist.Size()-1] = newprey;
            }
            if (preylist[i].death == true){
                preydeaths.SetBounds(0, preydeaths.Size());
                preydeaths[preydeaths.Size()-1] = i;
            }
        }
        // Update predator list with new predator list and deaths
        if (preddeaths.Size() > 0){
            for (int i = 0; i < preddeaths.Size(); i++){
                predlist.RemoveItem(preddeaths[i]);
                predlist.SetBounds(0, predlist.Size()-2);
            }
        }
        if (newpredlist.Size() > 0){
            for (int i = 0; i < newpredlist.Size(); i++){
                predlist.SetBounds(0, predlist.Size());
                predlist[predlist.Size()-1] = newpredlist[i];
            }
        }
        // Update prey list with new prey list and deaths
        if (preydeaths.Size() > 0){
            for (int i = 0; i < preydeaths.Size(); i++){
                preylist.RemoveItem(preydeaths[i]);
                preylist.SetBounds(0, preylist.Size()-2);
            }
        }
        if (newpreylist.Size() > 0){
            for (int i = 0; i < newpreylist.Size(); i++){
                preylist.SetBounds(0, preylist.Size());
                preylist[preylist.Size()-1] = newpreylist[i];
            }
        }
        // Save
        preyfile << prey_pos << endl;
        preypopfile << preylist.Size() << " ";
        predfile << pred_pos << endl;
        predpopfile << predlist.Size() << " ";
        foodfile << food_pos << endl;
        double foodsum = 0.0;
        for (int i = 0; i < food_pos.Size(); i++){
            foodsum += WorldFood[food_pos[i]];
        }
        foodpopfile << foodsum << " ";
        // Check Population Collapse
        if (preylist.Size() <= 0 || predlist.Size() <= 0){
            break;
        }
        else{
            newpreylist.~TVector();
            preydeaths.~TVector();
            newpredlist.~TVector();
            preddeaths.~TVector();
            prey_pos.~TVector();
            pred_pos.~TVector();
            dead_food.~TVector();
        }
    }
    preyfile.close();
    preypopfile.close();
    predfile.close();
    predpopfile.close();
	foodfile.close();
    foodpopfile.close();

    return 0;
}

// // ------------------------------------
// // Interaction Rate Data Functions
// // ------------------------------------
// void DeriveLambdaH(Prey &prey, Predator &predator, RandomState &rs, double &maxCC, double &maxprey, int &samplesize, double &transient)
// {
//     ofstream lambHfile("menagerie/IndBatch2/analysis_results/ns_15/lambH.dat");
//     // for (int i = 0; i <= 0; i++){
//         TVector<TVector<double> > lambH;
//         for (int j = 0; j <= maxCC; j++){
//             TVector<double> lambHcc;
//             for (int k = 0; k <= samplesize; k++){
//                 int carrycapacity = j;
//                 // Fill World to Carrying Capacity
//                 TVector<double> food_pos;
//                 TVector<double> WorldFood(1, SpaceSize);
//                 WorldFood.FillContents(0.0);
//                 for (int i = 0; i <= carrycapacity; i++){
//                     int f = rs.UniformRandomInteger(1,SpaceSize);
//                     WorldFood[f] = 1.0;
//                     food_pos.SetBounds(0, food_pos.Size());
//                     food_pos[food_pos.Size()-1] = f;
//                 }
//                 // Seed preylist with starting population
//                 TVector<Prey> preylist(0,0);
//                 TVector<double> prey_pos;
//                 preylist[0] = prey;
//                 // for (int i = 0; i < j; i++){
//                 //     Prey newprey(prey_netsize, prey_gain, prey_s_width, prey_frate, prey_feff, prey_BT_metaloss, prey_b_thresh);
//                 //     newprey.Reset(rs.UniformRandomInteger(0,SpaceSize), 1.5);
//                 //     newprey.NervousSystem = prey.NervousSystem;
//                 //     newprey.sensorweights = prey.sensorweights;
//                 //     preylist.SetBounds(0, preylist.Size());
//                 //     preylist[preylist.Size()-1] = newprey;
//                 //     }
//                 // Make dummy predator list
//                 TVector<double> pred_pos(0,-1);
//                 double munch_count = 0;
//                 for (double time = 0; time < RateDuration; time += BTStepSize){
//                     // Remove chomped food from food list
//                     TVector<double> dead_food(0,-1);
//                     for (int i = 0; i < food_pos.Size(); i++){
//                         if (WorldFood[food_pos[i]] <= 0){
//                             dead_food.SetBounds(0, dead_food.Size());
//                             dead_food[dead_food.Size()-1] = food_pos[i];
//                         }
//                     }
//                     if (dead_food.Size() > 0){
//                         for (int i = 0; i < dead_food.Size(); i++){
//                             food_pos.RemoveFood(dead_food[i]);
//                             food_pos.SetBounds(0, food_pos.Size()-2);
//                         }
//                     }
//                     // Carrying capacity is 0 indexed, add 1 for true amount
//                     for (int i = 0; i < ((carrycapacity+1) - food_pos.Size()); i++){
//                         double c = rs.UniformRandom(0,1);
//                         if (c <= BT_G_Rate){
//                             int f = rs.UniformRandomInteger(1,SpaceSize);
//                             WorldFood[f] = 1.0;
//                             food_pos.SetBounds(0, food_pos.Size());
//                             food_pos[food_pos.Size()-1] = f;
//                         }
//                     }
//                     for (int i = 0; i < preylist.Size(); i++){
//                         // Prey Sense & Step
//                         preylist[i].Sense(food_pos, pred_pos);
//                         preylist[i].Step(BTStepSize, WorldFood);
//                         // Check Births
//                         if (preylist[i].birth == true){
//                             preylist[i].state = preylist[i].state - prey_repo;
//                             preylist[i].birth = false;
//                         }
//                         // Check Deaths
//                         if (preylist[i].death == true){
//                             preylist[i].Reset(rs.UniformRandomInteger(0,SpaceSize), 2.0);
//                             preylist[i].death = false;
//                         }
//                         // Check # of times food crossed
//                         if (time > transient){
//                             munch_count += preylist[i].snackflag;
//                             preylist[i].snackflag = 0.0;
//                         }
//                     }
//                 }
//                 double munchrate = munch_count/((RateDuration-transient)/BTStepSize);
//                 lambHcc.SetBounds(0, lambHcc.Size());
//                 lambHcc[lambHcc.Size()-1] = munchrate;
//             }
//         //     lambH.SetBounds(0, lambH.Size());
//         //     lambH[lambH.Size()-1] = lambHcc;
//         // }
//         lambHfile << lambHcc << endl;
//         lambHcc.~TVector();
//     }
//     // Save
//     lambHfile.close();
// }

// void DeriveLambdaH2(Prey &prey, Predator &predator, RandomState &rs, double &maxCC, double &maxprey, int &samplesize, double &transient)
// {
//     ofstream lambHfile("menagerie/IndBatch2/analysis_results/ns_15/lambH3.dat");
//     // for (int i = 0; i <= 0; i++){
//         TVector<TVector<double> > lambH;
//         for (int j = -1; j <= maxprey; j++){
//             TVector<double> lambHcc;
//             for (int k = 0; k <= samplesize; k++){
//                 int carrycapacity = 29;
//                 // Fill World to Carrying Capacity
//                 TVector<double> food_pos;
//                 TVector<double> WorldFood(1, SpaceSize);
//                 WorldFood.FillContents(0.0);
//                 for (int i = 0; i <= carrycapacity; i++){
//                     int f = rs.UniformRandomInteger(1,SpaceSize);
//                     WorldFood[f] = 1.0;
//                     food_pos.SetBounds(0, food_pos.Size());
//                     food_pos[food_pos.Size()-1] = f;
//                 }
//                 // Seed preylist with starting population
//                 TVector<Prey> preylist(0,0);
//                 TVector<double> prey_pos;
//                 preylist[0] = prey;
//                 for (int i = 0; i < j; i++){
//                     Prey newprey(prey_netsize, prey_gain, prey_s_width, prey_frate, prey_feff, prey_BT_metaloss, prey_b_thresh);
//                     newprey.Reset(rs.UniformRandomInteger(0,SpaceSize), 1.5);
//                     newprey.NervousSystem = prey.NervousSystem;
//                     newprey.sensorweights = prey.sensorweights;
//                     preylist.SetBounds(0, preylist.Size());
//                     preylist[preylist.Size()-1] = newprey;
//                     }
//                 // Make dummy predator list
//                 TVector<double> pred_pos(0,-1);
//                 double munch_count = 0;
//                 for (double time = 0; time < RateDuration; time += BTStepSize){
//                     // Remove chomped food from food list
//                     TVector<double> dead_food(0,-1);
//                     for (int i = 0; i < food_pos.Size(); i++){
//                         if (WorldFood[food_pos[i]] <= 0){
//                             dead_food.SetBounds(0, dead_food.Size());
//                             dead_food[dead_food.Size()-1] = food_pos[i];
//                         }
//                     }
//                     if (dead_food.Size() > 0){
//                         for (int i = 0; i < dead_food.Size(); i++){
//                             food_pos.RemoveFood(dead_food[i]);
//                             food_pos.SetBounds(0, food_pos.Size()-2);
//                         }
//                     }
//                     // Carrying capacity is 0 indexed, add 1 for true amount
//                     for (int i = 0; i < ((carrycapacity+1) - food_pos.Size()); i++){
//                         double c = rs.UniformRandom(0,1);
//                         if (c <= BT_G_Rate){
//                             int f = rs.UniformRandomInteger(1,SpaceSize);
//                             WorldFood[f] = 1.0;
//                             food_pos.SetBounds(0, food_pos.Size());
//                             food_pos[food_pos.Size()-1] = f;
//                         }
//                     }
//                     for (int i = 0; i < preylist.Size(); i++){
//                         // Prey Sense & Step
//                         preylist[i].Sense(food_pos, pred_pos);
//                         preylist[i].Step(BTStepSize, WorldFood);
//                         // Check Births
//                         if (preylist[i].birth == true){
//                             preylist[i].state = preylist[i].state - prey_repo;
//                             preylist[i].birth = false;
//                         }
//                         // Check Deaths
//                         if (preylist[i].death == true){
//                             preylist[i].Reset(rs.UniformRandomInteger(0,SpaceSize), 2.0);
//                             preylist[i].death = false;
//                         }
//                         // Check # of times food crossed
//                         if (time > transient){
//                             munch_count += preylist[i].snackflag;
//                             preylist[i].snackflag = 0.0;
//                         }
//                     }
//                 }
//                 double munchrate = munch_count/((RateDuration-transient)/BTStepSize);
//                 lambHcc.SetBounds(0, lambHcc.Size());
//                 lambHcc[lambHcc.Size()-1] = munchrate;
//             }
//         //     lambH.SetBounds(0, lambH.Size());
//         //     lambH[lambH.Size()-1] = lambHcc;
//         // }
//         lambHfile << lambHcc << endl;
//         lambHcc.~TVector();
//     }
//     // Save
//     lambHfile.close();
// }

// void DeriveRR(RandomState &rs, double &testCC, int &samplesize)
// {
//     ofstream RRfile("menagerie/IndBatch2/analysis_results/ns_15/RR.dat");
//     for (int r = -1; r <= testCC; r++){
//         TVector<double> RR;
//         for (int k = 0; k <= samplesize; k++){
//             double counter = 0;
//             for (double time = 0; time < RateDuration; time += BTStepSize){
//                 for (int i = 0; i < ((testCC+1) - (r+1)); i++){
//                     double c = rs.UniformRandom(0,1);
//                     if (c <= BT_G_Rate){
//                         int f = rs.UniformRandomInteger(1,SpaceSize);
//                         counter += 1;
//                     }
//                 }
//             }
//             RR.SetBounds(0, RR.Size());
//             RR[RR.Size()-1] = counter/(RateDuration/BTStepSize);
//         }
//         RRfile << RR << endl;
//         RR.~TVector();
//     }
//     // Save
//     RRfile.close();
// }

// void DeriveLambdaP(Prey &prey, Predator &predator, RandomState &rs, double &maxprey, int &samplesize, double &transient)
// {
//     ofstream lambCfile("menagerie/IndBatch2/analysis_results/ns_15/lambC.dat");
//     for (int j = -1; j<=maxprey; j++)
//     {   
//         TVector<double> lambC;
//         for (int k = 0; k<=samplesize; k++){
//             // Fill World to Carrying Capacity
//             TVector<double> food_pos;
//             TVector<double> WorldFood(1, SpaceSize);
//             WorldFood.FillContents(0.0);
//             for (int i = 0; i <= CC; i++){
//                 int f = rs.UniformRandomInteger(1,SpaceSize);
//                 WorldFood[f] = 1.0;
//                 food_pos.SetBounds(0, food_pos.Size());
//                 food_pos[food_pos.Size()-1] = f;
//             }
//             // Seed preylist with starting population
//             TVector<Prey> preylist(0,0);
//             TVector<double> prey_pos;
//             preylist[0] = prey;
//             for (int i = 0; i < j; i++){
//                 Prey newprey(prey_netsize, prey_gain, prey_s_width, prey_frate, prey_feff, prey_BT_metaloss, prey_b_thresh);
//                 newprey.Reset(rs.UniformRandomInteger(0,SpaceSize), 1.5);
//                 newprey.NervousSystem = prey.NervousSystem;
//                 newprey.sensorweights = prey.sensorweights;
//                 preylist.SetBounds(0, preylist.Size());
//                 preylist[preylist.Size()-1] = newprey;
//                 }
//             double munch_count = 0;
//             for (double time = 0; time < RateDuration; time += BTStepSize){
//                 // Remove chomped food from food list
//                 TVector<double> dead_food(0,-1);
//                 for (int i = 0; i < food_pos.Size(); i++){
//                     if (WorldFood[food_pos[i]] <= 0){
//                         dead_food.SetBounds(0, dead_food.Size());
//                         dead_food[dead_food.Size()-1] = food_pos[i];
//                     }
//                 }
//                 if (dead_food.Size() > 0){
//                     for (int i = 0; i < dead_food.Size(); i++){
//                         food_pos.RemoveFood(dead_food[i]);
//                         food_pos.SetBounds(0, food_pos.Size()-2);
//                     }
//                 }
//                 // Carrying capacity is 0 indexed, add 1 for true amount
//                 for (int i = 0; i < ((CC+1) - food_pos.Size()); i++){
//                     double c = rs.UniformRandom(0,1);
//                     if (c <= BT_G_Rate){
//                         int f = rs.UniformRandomInteger(1,SpaceSize);
//                         WorldFood[f] = 1.0;
//                         food_pos.SetBounds(0, food_pos.Size());
//                         food_pos[food_pos.Size()-1] = f;
//                     }
//                 }
//                 // Prey Sense & Step
//                 TVector<Prey> newpreylist;
//                 TVector<int> deaths;
//                 TVector<double> prey_pos;
//                 TVector<double> pred_pos;
//                 pred_pos.SetBounds(0, pred_pos.Size());
//                 pred_pos[pred_pos.Size()-1] = predator.pos;
//                 for (int i = 0; i < preylist.Size(); i++){
//                     if (preylist[i].death == true){
//                         preylist[i].Reset(rs.UniformRandomInteger(0,SpaceSize), 1.5);
//                     }
//                     else{
//                         preylist[i].Sense(food_pos, pred_pos);
//                         preylist[i].Step(BTStepSize, WorldFood);
//                     }
//                 }
//                 for (int i = 0; i <= preylist.Size()-1; i++){
//                     prey_pos.SetBounds(0, prey_pos.Size());
//                     prey_pos[prey_pos.Size()-1] = preylist[i].pos;
//                 }

//                 // Predator Sense & Step
//                 predator.Sense(prey_pos);
//                 predator.Step(BTStepSize, WorldFood, preylist);
//                 // Check # of times food crossed
//                 if(time > transient){
//                     munch_count += predator.snackflag;
//                     predator.snackflag = 0.0;
//                 }
//             }

//             double munchrate = munch_count/((RateDuration-transient)/BTStepSize);
//             lambC.SetBounds(0, lambC.Size());
//             lambC[lambC.Size()-1] = munchrate;
//         }
//         lambCfile << lambC << endl;
//         lambC.~TVector();
//     }
//     // Save
//     lambCfile.close();
// }

// // void CollectEcoRates(TVector<double> &genotype, RandomState &rs)
// {
//     ofstream erates("menagerie/IndBatch2/analysis_results/ns_15/ecosystem_rates.dat");
//     // Translate to phenotype
// 	TVector<double> phenotype;
// 	phenotype.SetBounds(1, VectSize);
// 	GenPhenMapping(genotype, phenotype);
//     // Create agents
//     Prey Agent1(prey_netsize, prey_gain, prey_s_width, prey_frate, prey_feff, prey_BT_metaloss, prey_b_thresh);
//     Predator Agent2(pred_gain, pred_s_width, pred_frate, pred_BT_handling_time);
//     Agent2.condition = pred_condition;
//     // Set nervous system
//     Agent1.NervousSystem.SetCircuitSize(prey_netsize);
//     int k = 1;
//     // Prey Time-constants
//     for (int i = 1; i <= prey_netsize; i++) {
//         Agent1.NervousSystem.SetNeuronTimeConstant(i,phenotype(k));
//         k++;
//     }
//     // Prey Biases
//     for (int i = 1; i <= prey_netsize; i++) {
//         Agent1.NervousSystem.SetNeuronBias(i,phenotype(k));
//         k++;
//     }
//     // Prey Neural Weights
//     for (int i = 1; i <= prey_netsize; i++) {
//         for (int j = 1; j <= prey_netsize; j++) {
//             Agent1.NervousSystem.SetConnectionWeight(i,j,phenotype(k));
//             k++;
//         }
//     }
//     // Prey Sensor Weights
//     for (int i = 1; i <= prey_netsize*3; i++) {
//         Agent1.sensorweights[i] = phenotype(k);
//         k++;
//     }
//     // Save Growth Rates
//     // Max growth rate of producers is the chance of a new plant coming in on a given time step
//     double systemcc = CC+1; // 0 indexed
//     double rr = BT_G_Rate;
//     erates << rr << " ";
//     erates << systemcc << " ";
//     erates << Agent1.frate << " ";
//     erates << Agent1.feff << " ";
//     erates << Agent1.metaloss << " ";
//     erates << Agent2.frate << " ";
//     // erates << Agent2.feff << " ";
//     // erates << Agent2.metaloss << " ";
//     erates.close();

//     // Set Sampling Range & Frequency
//     double maxCC = 300;
//     double maxprey = 60;
//     double transient = 100.0;
//     int samplesize = 10;
//     double testCC = 29;
//     // Collect rr at testCC
//     printf("Collecting Growth Rate at Test Carrying Capacity\n");
//     DeriveRR(rs, testCC, samplesize);
//     // Collect Prey Lambda & r
//     printf("Collecting Prey rates\n");
//     DeriveLambdaH(Agent1, Agent2, rs, maxCC, maxprey, samplesize, transient);
//     DeriveLambdaH2(Agent1, Agent2, rs, maxCC, maxprey, samplesize, transient);
//     // Collect Predator Lambda & r
//     // printf("Collecting Predator rates\n");
//     // DeriveLambdaP(Agent1, Agent2, rs, maxprey, samplesize, transient);
// }

// ------------------------------------
// Sensory Sample Functions
// ------------------------------------
double SSCoexist(TVector<double> &prey_genotype, TVector<double> &pred_genotype, RandomState &rs, int agent) 
{
    // Start output files
    std::string PreySSfile("menagerie/TestBatch/analysis_results/ns_");
    std::string PredSSfile("menagerie/TestBatch/analysis_results/ns_");
    std::string pyfile = "menagerie/TestBatch/analysis_results/ns_";
    std::string pdfile = "menagerie/TestBatch/analysis_results/ns_";
    std::string ffile = "menagerie/TestBatch/analysis_results/ns_";
    std::string fpfile = "menagerie/TestBatch/analysis_results/ns_";
    PreySSfile += std::to_string(agent);
    PredSSfile += std::to_string(agent);
    pyfile += std::to_string(agent);
    pdfile += std::to_string(agent);
    ffile += std::to_string(agent);
    fpfile += std::to_string(agent);
    PreySSfile += "/Prey_SenS.dat";
    PredSSfile += "/Pred_SenS.dat";
    pyfile += "/prey_pos.dat";
    pdfile += "/pred_pos.dat";
    ffile += "/food_pos.dat";
    fpfile += "/food_pop.dat";
    ofstream preySS(PreySSfile);
    ofstream predSS(PredSSfile);
    ofstream preyfile(pyfile);
    ofstream predfile(pdfile);
    ofstream foodfile(ffile);
    ofstream foodpopfile(fpfile);

    TVector<double> prey_FS;
    TVector<double> prey_PS;
    TVector<double> prey_SS;
    TVector<double> prey_NO1;
    TVector<double> prey_N1FS;
    TVector<double> prey_N1PS;
    TVector<double> prey_N1SS;
    TVector<double> prey_NO2;
    TVector<double> prey_N2FS;
    TVector<double> prey_N2PS;
    TVector<double> prey_N2SS;
    TVector<double> prey_NO3;
    TVector<double> prey_N3FS;
    TVector<double> prey_N3PS;
    TVector<double> prey_N3SS;
    TVector<double> prey_mov;

    TVector<double> pred_PS;
    TVector<double> pred_SS;
    TVector<double> pred_NO1;
    TVector<double> pred_N1FS;
    TVector<double> pred_N1PS;
    TVector<double> pred_N1SS;
    TVector<double> pred_NO2;
    TVector<double> pred_N2FS;
    TVector<double> pred_N2PS;
    TVector<double> pred_N2SS;
    TVector<double> pred_NO3;
    TVector<double> pred_N3FS;
    TVector<double> pred_N3PS;
    TVector<double> pred_N3SS;
    TVector<double> pred_mov;

    // Set running outcome
    double outcome = 99999999999.0;
    // Translate to phenotypes
	TVector<double> prey_phenotype;
	TVector<double> pred_phenotype;
	prey_phenotype.SetBounds(1, PreyVectSize);
    pred_phenotype.SetBounds(1, PredVectSize);
	GenPhenMapping(prey_genotype, prey_phenotype, 0);
	GenPhenMapping(pred_genotype, pred_phenotype, 1);
    // Create agents
    Prey Agent1(prey_netsize, prey_gain, prey_s_width, prey_frate, prey_feff, prey_metaloss, prey_b_thresh);
    Predator Agent2(pred_netsize, pred_gain, pred_s_width, pred_frate, pred_feff, pred_metaloss, pred_b_thresh, pred_handling_time);
    // Set Prey nervous system
    Agent1.NervousSystem.SetCircuitSize(prey_netsize);
    Agent2.NervousSystem.SetCircuitSize(pred_netsize);
    int k = 1;
    // Prey Time-constants
    for (int i = 1; i <= prey_netsize; i++) {
        Agent1.NervousSystem.SetNeuronTimeConstant(i,prey_phenotype(k));
        k++;
    }
    // Prey Biases
    for (int i = 1; i <= prey_netsize; i++) {
        Agent1.NervousSystem.SetNeuronBias(i,prey_phenotype(k));
        k++;
    }
    // Prey Neural Weights
    for (int i = 1; i <= prey_netsize; i++) {
        for (int j = 1; j <= prey_netsize; j++) {
            Agent1.NervousSystem.SetConnectionWeight(i,j,prey_phenotype(k));
            k++;
        }
    }
    int j = k;
    // Prey Sensor Weights
    for (int i = 1; i <= prey_netsize*3; i++) {
        Agent1.sensorweights[i] = prey_phenotype(k);
        k++;
    }
    Agent1.s_scalar = prey_phenotype(k);
    k++;
    Agent1.a_scalar = prey_phenotype(k);
    k++;

    k=1;
    // Predator Time-constants
    for (int i = 1; i <= pred_netsize; i++) {
        Agent2.NervousSystem.SetNeuronTimeConstant(i,pred_phenotype(k));
        k++;
    }
    // Predator Biases
    for (int i = 1; i <= pred_netsize; i++) {
        Agent2.NervousSystem.SetNeuronBias(i,pred_phenotype(k));
        k++;
    }
    // Predator Neural Weights
    for (int i = 1; i <= pred_netsize; i++) {
        for (int j = 1; j <= pred_netsize; j++) {
            Agent2.NervousSystem.SetConnectionWeight(i,j,pred_phenotype(k));
            k++;
        }
    }
    j = k;
    // Predator Sensor Weights
    for (int i = 1; i <= pred_netsize*2; i++) {
        Agent2.sensorweights[i] = pred_phenotype(k);
        k++;
    }
    Agent2.s_scalar = pred_phenotype(k);
    k++;
    Agent2.a_scalar = pred_phenotype(k);
    k++;

    // Run Simulation
    // Reset Agents & Vectors
    Agent1.Reset(rs.UniformRandomInteger(0,SpaceSize), 1.5);
    Agent2.Reset(rs.UniformRandomInteger(0,SpaceSize), 2.5);
    // Seed preylist with starting population
    TVector<Prey> preylist(0,0);
    preylist[0] = Agent1;
    for (int i = 0; i < start_prey; i++){
        Prey newprey(prey_netsize, prey_gain, prey_s_width, prey_frate, prey_feff, prey_BT_metaloss, prey_b_thresh);
        newprey.Reset(rs.UniformRandomInteger(0,SpaceSize), 1.5);
        newprey.NervousSystem = Agent1.NervousSystem;
        newprey.sensorweights = Agent1.sensorweights;
        preylist.SetBounds(0, preylist.Size());
        preylist[preylist.Size()-1] = newprey;
    }
    // Seed predlist with starting population
    TVector<Predator> predlist(0,0);
    predlist[0] = Agent2;
    for (int i = 0; i < start_pred; i++){
        Predator newpred(pred_netsize, pred_gain, pred_s_width, pred_frate, pred_feff, pred_BT_metaloss, pred_b_thresh, pred_BT_handling_time);
        newpred.Reset(rs.UniformRandomInteger(0,SpaceSize), 2.5);
        newpred.NervousSystem = Agent2.NervousSystem;
        newpred.sensorweights = Agent2.sensorweights;
        predlist.SetBounds(0, predlist.Size());
        predlist[predlist.Size()-1] = newpred;
    }
    // Fill World to Carrying Capacity
    TVector<double> food_pos(0,-1);
    TVector<double> WorldFood(1, SpaceSize);
    WorldFood.FillContents(0.0);
    for (int i = 0; i <= CC; i++){
        int f = rs.UniformRandomInteger(1,SpaceSize);
        WorldFood[f] = 1.0;
        food_pos.SetBounds(0, food_pos.Size());
        food_pos[food_pos.Size()-1] = f;
    }
    // Run Simulation
    for (double time = 0; time < PlotDuration; time += BTStepSize){
        // Remove chomped food from food list
        TVector<double> dead_food(0,-1);
        for (int i = 0; i < food_pos.Size(); i++){
            if (WorldFood[food_pos[i]] <= 0){
                dead_food.SetBounds(0, dead_food.Size());
                dead_food[dead_food.Size()-1] = food_pos[i];
            }
        }
        if (dead_food.Size() > 0){
            for (int i = 0; i < dead_food.Size(); i++){
                food_pos.RemoveFood(dead_food[i]);
                food_pos.SetBounds(0, food_pos.Size()-2);
            }
        }
        // Carrying capacity is 0 indexed, add 1 for true amount
        for (int i = 0; i < ((CC+1) - food_pos.Size()); i++){
                double c = rs.UniformRandom(0,1);
                if (c <= BT_G_Rate){
                    int f = rs.UniformRandomInteger(1,SpaceSize);
                    WorldFood[f] = 1.0;
                    food_pos.SetBounds(0, food_pos.Size());
                    food_pos[food_pos.Size()-1] = f;
                }
            }
        // Update Prey Positions
        TVector<double> prey_pos;
        for (int i = 0; i < preylist.Size(); i++){
            prey_pos.SetBounds(0, prey_pos.Size());
            prey_pos[prey_pos.Size()-1] = preylist[i].pos;
        }
        // Predator Sense & Step
        TVector<Predator> newpredlist;
        TVector<int> preddeaths;
        for (int i = 0; i < predlist.Size(); i++){
            predlist[i].Sense(prey_pos);
            pred_PS.SetBounds(0, pred_PS.Size());
            pred_PS[pred_PS.Size()-1] = predlist[i].sensor;
            pred_N1PS.SetBounds(0, pred_N1PS.Size());
            pred_N1PS[pred_N1PS.Size()-1] = predlist[i].sensor * predlist[i].sensorweights[1];
            pred_N2PS.SetBounds(0, pred_N2PS.Size());
            pred_N2PS[pred_N2PS.Size()-1] = predlist[i].sensor * predlist[i].sensorweights[3];
            pred_N3PS.SetBounds(0, pred_N3PS.Size());
            pred_N3PS[pred_N3PS.Size()-1] = predlist[i].sensor * predlist[i].sensorweights[5];
            pred_SS.SetBounds(0, pred_SS.Size());
            pred_SS[pred_SS.Size()-1] = predlist[i].state;
            pred_N1SS.SetBounds(0, pred_N1SS.Size());
            pred_N1SS[pred_N1SS.Size()-1] = predlist[i].state * predlist[i].sensorweights[2];
            pred_N2SS.SetBounds(0, pred_N2SS.Size());
            pred_N2SS[pred_N2SS.Size()-1] = predlist[i].state * predlist[i].sensorweights[4];
            pred_N3SS.SetBounds(0, pred_N3SS.Size());
            pred_N3SS[pred_N3SS.Size()-1] = predlist[i].state * predlist[i].sensorweights[6];

            predlist[i].Step(BTStepSize, WorldFood, preylist);
            pred_NO1.SetBounds(0, pred_NO1.Size());
            pred_NO1[pred_NO1.Size()-1] = predlist[i].NervousSystem.NeuronOutput(1);
            pred_NO2.SetBounds(0, pred_NO2.Size());
            pred_NO2[pred_NO2.Size()-1] = predlist[i].NervousSystem.NeuronOutput(2);
            pred_NO3.SetBounds(0, pred_NO3.Size());
            pred_NO3[pred_NO3.Size()-1] = predlist[i].NervousSystem.NeuronOutput(3);
            pred_mov.SetBounds(0, pred_mov.Size());
            pred_mov[pred_mov.Size()-1] = (predlist[i].NervousSystem.NeuronOutput(2) - predlist[i].NervousSystem.NeuronOutput(1));
            
            if (predlist[i].birth == true){
                predlist[i].state = predlist[i].state - pred_repo;
                predlist[i].birth = false;
                // FOR POPS ONLY
                // Predator newpred(pred_netsize, pred_gain, pred_s_width, pred_frate, pred_feff, pred_BT_metaloss, pred_b_thresh, pred_BT_handling_time);
                // newpred.Reset(predlist[i].pos+2, pred_repo);
                // newpred.NervousSystem = Agent2.NervousSystem;
                // newpred.sensorweights = Agent2.sensorweights;
                // newpredlist.SetBounds(0, predlist.Size());
                // newpredlist[predlist.Size()-1] = newpred;
            }
            if (predlist[i].death == true){
                preddeaths.SetBounds(0, preddeaths.Size());
                preddeaths[preddeaths.Size()-1] = i;
            }
        }
        // Update Predator Positions
        TVector<double> pred_pos;
        for (int i = 0; i < predlist.Size(); i++){
            pred_pos.SetBounds(0, pred_pos.Size());
            pred_pos[pred_pos.Size()-1] = predlist[i].pos;
        }
        // Prey Sense & Step
        TVector<Prey> newpreylist;
        TVector<int> preydeaths;
        for (int i = 0; i < preylist.Size(); i++){
            preylist[i].Sense(food_pos, pred_pos);
            prey_FS.SetBounds(0, prey_FS.Size());
            prey_FS[prey_FS.Size()-1] = preylist[i].f_sensor;
            prey_N1FS.SetBounds(0, prey_N1FS.Size());
            prey_N1FS[prey_N1FS.Size()-1] = preylist[i].f_sensor * preylist[i].sensorweights[1];
            prey_N2FS.SetBounds(0, prey_N2FS.Size());
            prey_N2FS[prey_N2FS.Size()-1] = preylist[i].f_sensor * preylist[i].sensorweights[4];
            prey_N3FS.SetBounds(0, prey_N3FS.Size());
            prey_N3FS[prey_N3FS.Size()-1] = preylist[i].f_sensor * preylist[i].sensorweights[7];
            prey_PS.SetBounds(0, prey_PS.Size());
            prey_PS[prey_PS.Size()-1] = preylist[i].p_sensor;
            prey_N1PS.SetBounds(0, prey_N1PS.Size());
            prey_N1PS[prey_N1PS.Size()-1] = preylist[i].p_sensor * preylist[i].sensorweights[2];
            prey_N2PS.SetBounds(0, prey_N2PS.Size());
            prey_N2PS[prey_N2PS.Size()-1] = preylist[i].p_sensor * preylist[i].sensorweights[5];
            prey_N3PS.SetBounds(0, prey_N3PS.Size());
            prey_N3PS[prey_N3PS.Size()-1] = preylist[i].p_sensor * preylist[i].sensorweights[8];
            prey_SS.SetBounds(0, prey_SS.Size());
            prey_SS[prey_SS.Size()-1] = preylist[i].state;
            prey_N1SS.SetBounds(0, prey_N1SS.Size());
            prey_N1SS[prey_N1SS.Size()-1] = preylist[i].state * preylist[i].sensorweights[3];
            prey_N2SS.SetBounds(0, prey_N2SS.Size());
            prey_N2SS[prey_N2SS.Size()-1] = preylist[i].state * preylist[i].sensorweights[6];
            prey_N3SS.SetBounds(0, prey_N3SS.Size());
            prey_N3SS[prey_N3SS.Size()-1] = preylist[i].state * preylist[i].sensorweights[9];

            preylist[i].Step(BTStepSize, WorldFood);
            prey_NO1.SetBounds(0, prey_NO1.Size());
            prey_NO1[prey_NO1.Size()-1] = preylist[i].NervousSystem.NeuronOutput(1);
            prey_NO2.SetBounds(0, prey_NO2.Size());
            prey_NO2[prey_NO2.Size()-1] = preylist[i].NervousSystem.NeuronOutput(2);
            prey_NO3.SetBounds(0, prey_NO3.Size());
            prey_NO3[prey_NO3.Size()-1] = preylist[i].NervousSystem.NeuronOutput(3);
            prey_mov.SetBounds(0, prey_mov.Size());
            prey_mov[prey_mov.Size()-1] = (preylist[i].NervousSystem.NeuronOutput(2) - preylist[i].NervousSystem.NeuronOutput(1));
            
            if (preylist[i].birth == true){
                preylist[i].state = preylist[i].state - prey_repo;
                preylist[i].birth = false;
                // FOR POPS ONLY
                // Prey newprey(prey_netsize, prey_gain, prey_s_width, prey_frate, prey_feff, prey_metaloss, prey_b_thresh);
                // newprey.NervousSystem = preylist[i].NervousSystem;
                // newprey.sensorweights = preylist[i].sensorweights;
                // newprey.Reset(preylist[i].pos+2, prey_repo);
                // newpreylist.SetBounds(0, newpreylist.Size());
                // newpreylist[newpreylist.Size()-1] = newprey;
            }
            if (preylist[i].death == true){
                preydeaths.SetBounds(0, preydeaths.Size());
                preydeaths[preydeaths.Size()-1] = i;
            }
        }
        // Update prey list with new prey list and deaths
        if (preydeaths.Size() > 0){
            for (int i = 0; i < preydeaths.Size(); i++){
                preylist.RemoveItem(preydeaths[i]);
                preylist.SetBounds(0, preylist.Size()-2);
            }
        }
        if (newpreylist.Size() > 0){
            for (int i = 0; i < newpreylist.Size(); i++){
                preylist.SetBounds(0, preylist.Size());
                preylist[preylist.Size()-1] = newpreylist[i];
            }
        }
        // Save
        preyfile << prey_pos << endl;
        predfile << pred_pos << endl;
        foodfile << food_pos << endl;
        double foodsum = 0.0;
        for (int i = 0; i < food_pos.Size(); i++){
            foodsum += WorldFood[food_pos[i]];
        }
        foodpopfile << foodsum << " ";
        // Check Population Collapse
        if (preylist.Size() <= 0){
            break;
        }
        else{
            newpreylist.~TVector();
            preydeaths.~TVector();
            newpredlist.~TVector();
            preddeaths.~TVector();
            prey_pos.~TVector();
            pred_pos.~TVector();
            dead_food.~TVector();
        }
    }
    preyfile.close();
    predfile.close();
	foodfile.close();
    foodpopfile.close();

    preySS << prey_FS << endl << prey_N1FS << endl << prey_N2FS << endl << prey_N3FS << endl;
    preySS << prey_PS << endl << prey_N1PS << endl << prey_N2PS << endl << prey_N3PS << endl;
    preySS << prey_SS << endl << prey_N1SS << endl << prey_N2SS << endl << prey_N3SS << endl; 
    preySS << prey_NO1 << endl << prey_NO2 << endl << prey_NO3 << endl << prey_mov << endl;

    predSS << pred_PS << endl << pred_N1PS << endl << pred_N2PS << endl << pred_N3PS << endl;
    predSS << pred_SS << endl << pred_N1SS << endl << pred_N2SS << endl << pred_N3SS << endl; 
    predSS << pred_NO1 << endl << pred_NO2 << endl << pred_NO3 << endl << pred_mov << endl;

    preySS.close();
    predSS.close();
    
    return 0;
}

// // ---------------------------------------
// // Test Function for Code Development
// // ---------------------------------------
// void NewEco(TVector<double> &genotype, RandomState &rs)
// {
//     double test_CC = 29;
//     double start_CC = 10;
//     double start_prey_sim = 15;
//     double test_frate = prey_frate;
//     double test_feff = prey_feff;
//     ofstream ppfile("menagerie/IndBatch2/analysis_results/ns_15/sim_prey_pop.dat");
//     ofstream fpfile("menagerie/IndBatch2/analysis_results/ns_15/sim_food_pop.dat");
//     // Translate to phenotype
// 	TVector<double> phenotype;
// 	phenotype.SetBounds(1, VectSize);
// 	GenPhenMapping(genotype, phenotype);
//     // Create agents
//     // Playing with feff, frate, metaloss
//     Prey Agent1(prey_netsize, prey_gain, prey_s_width, prey_frate, prey_feff, prey_BT_metaloss, prey_b_thresh);
//     // Set nervous system
//     Agent1.NervousSystem.SetCircuitSize(prey_netsize);
//     int k = 1;
//     // Prey Time-constants
//     for (int i = 1; i <= prey_netsize; i++) {
//         Agent1.NervousSystem.SetNeuronTimeConstant(i,phenotype(k));
//         k++;
//     }
//     // Prey Biases
//     for (int i = 1; i <= prey_netsize; i++) {
//         Agent1.NervousSystem.SetNeuronBias(i,phenotype(k));
//         k++;
//     }
//     // Prey Neural Weights
//     for (int i = 1; i <= prey_netsize; i++) {
//         for (int j = 1; j <= prey_netsize; j++) {
//             Agent1.NervousSystem.SetConnectionWeight(i,j,phenotype(k));
//             k++;
//         }
//     }
//     // Prey Sensor Weights
//     for (int i = 1; i <= prey_netsize*3; i++) {
//         Agent1.sensorweights[i] = phenotype(k);
//         k++;
//     }
//     // Fill World to Carrying Capacity
//     TVector<double> food_pos;
//     TVector<double> WorldFood(1, SpaceSize);
//     WorldFood.FillContents(0.0);
//     for (int i = 0; i <= start_CC; i++){
//         int f = rs.UniformRandomInteger(1,SpaceSize);
//         WorldFood[f] = 1.0;
//         food_pos.SetBounds(0, food_pos.Size());
//         food_pos[food_pos.Size()-1] = f;
//     }
//     // Make dummy predator list
//     TVector<double> pred_pos(0,-1);
//     TVector<Prey> preylist(0,0);
//     preylist[0] = Agent1;
//     // // Carrying capacity is 0 indexed, add 1 for true amount
//     // for (int i = 0; i < 200; i++){
//     //     double food_count = food_pos.Size();
//     //     double s_chance = 1 - food_count/(test_CC+1);
//     //     double c = rs.UniformRandom(0,1)*50;
//     //     if (c < s_chance){
//     //         int f = rs.UniformRandomInteger(1,SpaceSize);
//     //         WorldFood[f] = 1.0;
//     //         food_pos.SetBounds(0, food_pos.Size());
//     //         food_pos[food_pos.Size()-1] = f;
//     //     }
//     // }
//     for (int i = 0; i < start_prey_sim; i++){
//         Prey newprey(prey_netsize, prey_gain, prey_s_width, test_frate, test_feff, prey_BT_metaloss, prey_b_thresh);
//         newprey.Reset(rs.UniformRandomInteger(0,SpaceSize), 1.5);
//         newprey.NervousSystem = Agent1.NervousSystem;
//         newprey.sensorweights = Agent1.sensorweights;
//         preylist.SetBounds(0, preylist.Size());
//         preylist[preylist.Size()-1] = newprey;
//         }
//     for (double time = 0; time < PlotDuration*300; time += StepSize){
//         // Remove chomped food from food list
//         TVector<double> dead_food(0,-1);
//         for (int i = 0; i < food_pos.Size(); i++){
//             if (WorldFood[food_pos[i]] <= 0){
//                 dead_food.SetBounds(0, dead_food.Size());
//                 dead_food[dead_food.Size()-1] = food_pos[i];
//             }
//         }
//         if (dead_food.Size() > 0){
//             for (int i = 0; i < dead_food.Size(); i++){
//                 food_pos.RemoveFood(dead_food[i]);
//                 food_pos.SetBounds(0, food_pos.Size()-2);
//             }
//         }
//         // Carrying capacity is 0 indexed, add 1 for true amount
//         double c = rs.UniformRandom(0,1);
//         for (int i = 0; i < ((test_CC+1) - food_pos.Size()); i++){
//             double c = rs.UniformRandom(0,1);
//             if (c <= BT_G_Rate){
//                 int f = rs.UniformRandomInteger(1,SpaceSize);
//                 WorldFood[f] = 1.0;
//                 food_pos.SetBounds(0, food_pos.Size());
//                 food_pos[food_pos.Size()-1] = f;
//             }
//         }
//         // Prey Sense & Step
//         TVector<Prey> newpreylist;
//         TVector<int> preydeaths;
//         double total_state = 0;
//         for (int i = 0; i < preylist.Size(); i++){
//             preylist[i].Sense(food_pos, pred_pos);
//             preylist[i].Step(StepSize, WorldFood);
//             total_state += preylist[i].state;
//             if (preylist[i].birth == true){
//                 preylist[i].state = preylist[i].state - prey_repo;
//                 preylist[i].birth = false;
//                 Prey newprey(prey_netsize, prey_gain, prey_s_width, test_frate, test_feff, prey_BT_metaloss, prey_b_thresh);
//                 newprey.NervousSystem = preylist[i].NervousSystem;
//                 newprey.sensorweights = preylist[i].sensorweights;
//                 newprey.Reset(preylist[i].pos+2, prey_repo);
//                 newpreylist.SetBounds(0, newpreylist.Size());
//                 newpreylist[newpreylist.Size()-1] = newprey;
//             }
//             if (preylist[i].death == true){
//                 preydeaths.SetBounds(0, preydeaths.Size());
//                 preydeaths[preydeaths.Size()-1] = i;
//             }
//         }
//         // Update prey list with new prey list and deaths
//         if (preydeaths.Size() > 0){
//             for (int i = 0; i < preydeaths.Size(); i++){
//                 preylist.RemoveItem(preydeaths[i]);
//                 preylist.SetBounds(0, preylist.Size()-2);
//             }
//         }
//         if (newpreylist.Size() > 0){
//             for (int i = 0; i < newpreylist.Size(); i++){
//                 preylist.SetBounds(0, preylist.Size());
//                 preylist[preylist.Size()-1] = newpreylist[i];
//             }
//         }
//         ppfile << total_state << endl;
//         double total_food = 0;
//         for (int i = 0; i < WorldFood.Size();i++){
//             if (WorldFood[i] > 0){
//                 total_food += WorldFood[i];
//             }
//         }
//         fpfile << total_food << endl;
//         // Check Population Collapse
//         if (preylist.Size() <= 0){
//             break;
//         }
//         else{
//             newpreylist.~TVector();
//             preydeaths.~TVector();
//             dead_food.~TVector();
//         }
//     }
//     // Save
//     ppfile.close();
//     fpfile.close();
// }

// ================================================
// E. MAIN FUNCTION
// ================================================
int main (int argc, const char* argv[]) 
{
// ================================================
// EVOLUTION
// ================================================
	// long randomseed = static_cast<long>(time(NULL));
    // std::string fileIndex = "";  // Default empty string for file index
    // if (argc > 1) {
    //     randomseed += atoi(argv[1]);
    //     fileIndex = "_" + std::string(argv[1]); // Append the index to the file name
    // }

    // TSearch prey_s(PreyVectSize);
    // TSearch pred_s(PredVectSize);

    // std::ofstream preyevolfile;
    // std::string preyfilename = std::string("prey_evolutions") + "/evol" + fileIndex + ".dat"; // Create a unique file name
    // preyevolfile.open(preyfilename);
    // cout.rdbuf(preyevolfile.rdbuf());
    // std::ofstream predevolfile;
    // std::string predfilename = std::string("pred_evolutions") + "/evol" + fileIndex + ".dat"; // Create a unique file name
    // predevolfile.open(predfilename);
    // cout.rdbuf(predevolfile.rdbuf());
    // // Save the seed to a file
    // std::ofstream seedfile;
    // std::string seedfilename = std::string("seeds") + "/seed" + fileIndex + ".dat"; // Create a unique file name for seed
    // seedfile.open(seedfilename);
    // seedfile << randomseed << std::endl;
    // seedfile.close();
    // RandomState rs(randomseed);

    // // Configure the search
    // prey_s.SetRandomSeed(randomseed);
    // prey_s.SetSearchResultsDisplayFunction(ResultsDisplay);
    // prey_s.SetPopulationStatisticsDisplayFunction(EvolutionaryRunDisplay);
    // prey_s.SetSelectionMode(RANK_BASED);
    // prey_s.SetReproductionMode(CO_GENETIC_ALGORITHM);
    // prey_s.SetPopulationSize(PREY_POPSIZE);
    // prey_s.SetMaxGenerations(GENS);
    // prey_s.SetCrossoverProbability(CROSSPROB);
    // prey_s.SetCrossoverMode(UNIFORM);
    // prey_s.SetMutationVariance(MUTVAR);
    // prey_s.SetMaxExpectedOffspring(EXPECTED);
    // prey_s.SetElitistFraction(ELITISM);
    // prey_s.SetSearchConstraint(1);
    // prey_s.SetReEvaluationFlag(1);

    // pred_s.SetRandomSeed(randomseed);
    // pred_s.SetSearchResultsDisplayFunction(ResultsDisplay);
    // pred_s.SetPopulationStatisticsDisplayFunction(EvolutionaryRunDisplay);
    // pred_s.SetSelectionMode(RANK_BASED);
    // pred_s.SetReproductionMode(CO_GENETIC_ALGORITHM);
    // pred_s.SetPopulationSize(PRED_POPSIZE);
    // pred_s.SetMaxGenerations(GENS);
    // pred_s.SetCrossoverProbability(CROSSPROB);
    // pred_s.SetCrossoverMode(UNIFORM);
    // pred_s.SetMutationVariance(MUTVAR);
    // pred_s.SetMaxExpectedOffspring(EXPECTED);
    // pred_s.SetElitistFraction(ELITISM);
    // pred_s.SetSearchConstraint(1);
    // pred_s.SetReEvaluationFlag(1);

    // int CC = CC;
    // int maxvolley = 20;
    
    // for (int i = 0; i < maxvolley; i++){
    //     cout.rdbuf(preyevolfile.rdbuf());
    //     TVector<double> BOpred;
    //     BOpred.SetBounds(1, PredVectSize);
    //     // for (int i = 1; i <= BOpred.Size(); i++){
    //     //     BOpred[i] = 0.0;
    //     // }
    //     BOpred = pred_s.BestIndividual();
    //     prey_s.SetSearchTerminationFunction(IntTerminationFunction);
    //     prey_s.SetCoEvaluationFunction(PreyTest);
    //     prey_s.SetBO(BOpred);
    //     TVector<double> bestotherprey = prey_s.BestOther();
    //     // printf("BO CHECK pred best given\n");
    //     // for (int i = 0; i <= bestotherprey.Size(); i++){
    //     //     printf("%f \n", bestotherprey[i]);
    //     // }
    //     prey_s.ExecuteCoSearch(BOpred);
    //     TVector<double> bestpreyVector;
    //     ofstream BestPreyIndividualFile;
    //     TVector<double> preyphenotype;
    //     preyphenotype.SetBounds(1, PreyVectSize);
    //     // Save the genotype of the best individual
    //     // Use the global index in file names
    //     std::string bestpreyGenFilename = std::string("prey_genomes") + "/best.gen" + fileIndex + ".dat";
    //     std::string bestpreyNsFilename = std::string("prey_nerves") + "/best.ns" + fileIndex + ".dat";
    //     bestpreyVector = prey_s.BestIndividual();
    //     BestPreyIndividualFile.open(bestpreyGenFilename);
    //     BestPreyIndividualFile << bestpreyVector << endl;
    //     BestPreyIndividualFile.close();
    //     // Also show the best individual in the Circuit Model form
    //     BestPreyIndividualFile.open(bestpreyNsFilename);
    //     GenPhenMapping(bestpreyVector, preyphenotype, 0);
    //     BestPreyIndividualFile << preyphenotype << endl;
    //     BestPreyIndividualFile.close();

    //     // BO~TVector();
    //     cout.rdbuf(predevolfile.rdbuf());
    //     TVector<double> BOprey;
    //     BOprey.SetBounds(1, PreyVectSize);
        
    //     BOprey = prey_s.BestIndividual();

    //     pred_s.SetSearchTerminationFunction(IntTerminationFunction);
    //     pred_s.SetCoEvaluationFunction(PredTest);
    //     pred_s.SetBO(BOprey);
    //     TVector<double> bestotherpred = pred_s.BestOther();
    //     // printf("BO CHECK best prey given\n");
    //     // for (int i = 0; i <= bestotherpred.Size(); i++){
    //     //     printf("%f \n", bestotherpred[i]);
    //     // }
    //     pred_s.ExecuteCoSearch(BOprey);
    //     TVector<double> bestpredVector;
    //     ofstream BestPredIndividualFile;
    //     TVector<double> predphenotype;
    //     predphenotype.SetBounds(1, PredVectSize);
    //     // Save the genotype of the best individual
    //     // Use the global index in file names
    //     std::string bestpredGenFilename = std::string("pred_genomes") + "/best.gen" + fileIndex + ".dat";
    //     std::string bestpredNsFilename = std::string("pred_nerves") + "/best.ns" + fileIndex + ".dat";
    //     bestpredVector = pred_s.BestIndividual();
    //     BestPredIndividualFile.open(bestpredGenFilename);
    //     BestPredIndividualFile << bestpredVector << endl;
    //     BestPredIndividualFile.close();
    //     // Also show the best individual in the Circuit Model form
    //     BestPredIndividualFile.open(bestpredNsFilename);
    //     GenPhenMapping(bestpredVector, predphenotype, 1);
    //     BestPredIndividualFile << predphenotype << endl;
    //     BestPredIndividualFile.close();

    //     // Destroy Vectors
    //     BOpred.~TVector();
    //     BOprey.~TVector();
    //     bestotherprey.~TVector();
    //     bestotherpred.~TVector();
    //     bestpreyVector.~TVector();
    //     bestpredVector.~TVector();
    //     preyphenotype.~TVector();
    //     predphenotype.~TVector();
    // }

    // return 0;

// ================================================
// RUN ANALYSES
// ================================================

    // // SET LIST OF AGENTS TO ANALYZE HERE, NEED LIST SIZE FOR INIT
    // TVector<int> analylist(0,1);
    // analylist.InitializeContents(15);

    // // // Behavioral Traces // // 
    // for (int i = 0; i < analylist.Size(); i++){
    //     int agent = analylist[i];
    //     // load the seed
    //     ifstream seedfile;
    //     double seed;
    //     seedfile.open("seed.dat");
    //     seedfile >> seed;
    //     seedfile.close();
    //     // load best prey
    //     ifstream prey_genefile;
    //     // SET WHICH BATCH YOU'RE EVALUATING HERE, CHECK ANALYSIS FUNCTIONS FOR THE SAME
    //     prey_genefile.open("menagerie/TestBatch/prey_genomes/best.gen_%%.dat", agent);
    //     TVector<double> prey_genotype(1, PreyVectSize);
    //     prey_genefile >> prey_genotype;
    //     prey_genefile.close();
    //     // load best predator
    //     ifstream pred_genefile;
    //     // SET WHICH BATCH YOU'RE EVALUATING HERE, CHECK ANALYSIS FUNCTIONS FOR THE SAME
    //     pred_genefile.open("menagerie/TestBatch/pred_genomes/best.gen_%%.dat", agent);
    //     TVector<double> pred_genotype(1, PredVectSize);
    //     pred_genefile >> pred_genotype;
    //     pred_genefile.close();
    //     // set the seed
    //     RandomState rs(seed);
    //     BehavioralTracesCoexist(prey_genotype, pred_genotype, rs, agent);
    // }

    // ANALYSES FOR JUST ONE AGENT
    // load the seed
    ifstream seedfile;
    double seed;
    seedfile.open("seed.dat");
    seedfile >> seed;
    seedfile.close();
    // load best prey
    ifstream prey_genefile;
    // SET WHICH BATCH YOU'RE EVALUATING HERE, CHECK ANALYSIS FUNCTIONS FOR THE SAME
    prey_genefile.open("menagerie/TestBatch/prey_genomes/best.gen_1.dat");
    TVector<double> prey_genotype(1, PreyVectSize);
    prey_genefile >> prey_genotype;
    prey_genefile.close();
    // load best predator
    ifstream pred_genefile;
    // SET WHICH BATCH YOU'RE EVALUATING HERE, CHECK ANALYSIS FUNCTIONS FOR THE SAME
    pred_genefile.open("menagerie/TestBatch/pred_genomes/best.gen_1.dat");
    TVector<double> pred_genotype(1, PredVectSize);
    pred_genefile >> pred_genotype;
    pred_genefile.close();
    // set the seed
    RandomState rs(seed);

    SSCoexist(prey_genotype, pred_genotype, rs, 1);

    // // // Interaction Rate Collection // //
    // CollectEcoRates(genotype, rs);

    // // Sensory Sample Collection // //
    // SensorySample(genotype, rs);

    // // Code Testbed // // 
    // NewEco(genotype, rs);

    return 0;

}