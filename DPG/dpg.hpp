#include <fcntl.h>
#include <unistd.h>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iomanip>
#include <map>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <algorithm>
#include <cstdlib>
#include "tiles.cpp"

// RANDOM NUMBER _GENERATOR_
//std::default_random_engine __GENERATOR__;

// Variables definint the number of tiles and the array for state representation in terms of tiles
#define NUM_TILINGS 32						// Number of tilings
#define MEMORY_SIZE 8192  						// Number of parameters to theta, memory size
float GAUSS_VAR =1.0;							// Gaussian variance
#define _GAMMA_ 0.99							// Discount factor
#define mcar_factor 1.0
#define mcar_min_position -1.2
#define mcar_max_position  0.6
#define mcar_max_velocity  0.07	// the negative of this in the minimum velocity
#define mcar_goal_position 0.5
#define POS_EPS 1E-8

// All the deterministic policy gradient parameters
float DPG_W[MEMORY_SIZE];
float DPG_T[MEMORY_SIZE];
float DPG_V[MEMORY_SIZE];
float DPG_TDs;

float DPG_T_alpha = 0.000625;
float DPG_W_alpha = 0.00003125;
float DPG_V_alpha = 0.00003125;

int TILE_FEATURES[NUM_TILINGS];					// Feature representation (tile coding)
int TILE_FEATURES_NEXT[NUM_TILINGS];			// Fewaturesfor next state

// Function declarations
void buildTileFeatures(const int, float*);
void buildTileFeaturesNext(const int, float*);
void DPGUpdatesTiles(float a, float nA, float rt, bool FLAG, float = 1.0);	

///////////////////////////// TILES VERSION ////////////////////////////
float dotPTiles(float*);
float pickActionTiles();
float pickActionRandomTiles(const float);
float pickActionTilesNext();
float pickActionRandomTilesNext(const float);
float computeValueTiles();


// Profiles
void McarParamInit();
void MCarInit();					// initialize car state
void MCarStep(int a);				// update car state for given action
bool MCarAtGoal ();					// is car at goal?

const float eEffect = 0.001;
const float gEffect = 0.0025;

float mcar_position;
float mcar_velocity;

void McarParamInit(){
	return;
}

void MCarInit(){
	// This function generates a random initial state	
	mcar_position = (asin(-1.0)/3.0) - ((rand()/((double)RAND_MAX)) - 0.5)/5.0;
	mcar_velocity = 0.0 - ((rand()/((double)RAND_MAX)) - 0.5)/50.0;
}

void MCarStep(float a){
	// Take the action
	mcar_velocity += a*eEffect - gEffect*cos(3.0*mcar_position);
	if (mcar_velocity > mcar_max_velocity)
		mcar_velocity = mcar_max_velocity;

	if (mcar_velocity < -mcar_max_velocity)
		mcar_velocity = -mcar_max_velocity;

	mcar_position = mcar_position + (mcar_velocity/mcar_factor);
	if (mcar_position > mcar_max_position)
		mcar_position = mcar_max_position;

	if (mcar_position < mcar_min_position)
		mcar_position = mcar_min_position;

	if ((mcar_position == mcar_min_position) && (mcar_velocity<0.0))
		mcar_velocity = 0.0;
}

bool MCarAtGoal(){
	// Is the goal location reached
	return (mcar_position>=mcar_goal_position);
}


// Definitions
float dotPTiles(float* vec){
	float r = 0.0;
	for (int i = 0; i < NUM_TILINGS; i++)
		r += vec[TILE_FEATURES[i]];
	return r;
}

float dotPTilesNext(float* vec){
	float r = 0.0;
	for (int i = 0; i < NUM_TILINGS; i++){
		r += vec[TILE_FEATURES_NEXT[i]];
	}
	return r;
}

float computeValueTiles(float action, const int nextFLAG){
	if (!nextFLAG){
		float mu_theta	= dotPTiles(&DPG_T[0]);
		float nabla 	= dotPTiles(&DPG_W[0]);
		float Vvs 		= dotPTiles(&DPG_V[0]);
		return ((action - mu_theta)*nabla + Vvs);
	}else{
		float mu_theta	= dotPTilesNext(&DPG_T[0]);
		float nabla 	= dotPTilesNext(&DPG_W[0]);
		float Vvs 		= dotPTilesNext(&DPG_V[0]);
		return ((action - mu_theta)*nabla + Vvs);
	}
}

std::default_random_engine _GENERATOR_;
float pickActionRandomTiles(const float GAlphaFactor){
	float m = 0.0;
	for (int i = 0; i < NUM_TILINGS; i++){
		m = m + DPG_T[TILE_FEATURES[i]];
	}

	const float var = GAUSS_VAR*GAlphaFactor;

	// Gaussian Sampling
	//std::default_random_engine _GENERATOR_;
	std::normal_distribution<float> distribution(m,var);
	float action = distribution(_GENERATOR_);

	// Clipping 
	//std::cout<<"THIS "<<action<<std::endl;
	action = (action > 1.0) ? 1.0 : action;
	action = (action < -1.0) ? -1.0 : action;

	// Return action
	return (action);
}

float pickActionRandomTilesNext(const float GAlphaFactor){
	float m = 0.0;
	for (int i = 0; i < NUM_TILINGS; i++)
		m = m + DPG_T[TILE_FEATURES_NEXT[i]];

	const float var = GAUSS_VAR*GAlphaFactor;

	// Gaussian Sampling
	//std::default_random_engine _GENERATOR_;
	std::normal_distribution<float> distribution(m,var);
	float action = distribution(_GENERATOR_);

	// Clipping 
	action = (action > 1.0) ? 1.0 : action;
	action = (action < -1.0) ? -1.0 : action;

	// Return action
	return (action);
}

float pickActionTiles(){
	float m = 0.0;
	for (int i = 0; i < NUM_TILINGS; i++)
		m = m + DPG_T[TILE_FEATURES[i]];
	
	if (m < -1.0) return (-1.0);
	if (m > 1.0) return (1.0);
	return m;
}

float pickActionTilesNext(){
	float m = 0.0;
	for (int i = 0; i < NUM_TILINGS; i++)
		m = m + DPG_T[TILE_FEATURES_NEXT[i]];
	
	if (m < -1.0) return (-1.0);
	if (m > 1.0) return (1.0);
	return m;
}

void buildTileFeatures(const int numFeatures, float* features){
	/*	This function builds generation tile-coding representation of the input features.
	*/
	float state_vars[numFeatures];
	for (int i = 0; i < numFeatures; i++)
		state_vars[i] = *features++;
	tiles(&TILE_FEATURES[0],NUM_TILINGS,MEMORY_SIZE, state_vars, numFeatures);
}

void buildTileFeaturesNext(const int numFeatures, float* features){
	/*	This function builds generation tile-coding representation of the input features.
	*/
	float state_vars[numFeatures];
	for (int i = 0; i < numFeatures; i++)
		state_vars[i] = *features++;
	tiles(&TILE_FEATURES_NEXT[0],NUM_TILINGS,MEMORY_SIZE, state_vars, numFeatures);
}

void DPGUpdatesTiles(float a, float nA, float rt, bool FLAG, float fac){	
	// Compute the TD Error
	float tdError = rt + (1 - (int)FLAG)*_GAMMA_*computeValueTiles(nA, true) - computeValueTiles(a, false);

	// Update the parameters
	float temp = dotPTiles(&DPG_W[0]);
	const float T_temp = DPG_T_alpha*fac*temp;
	const float W_temp = DPG_W_alpha*fac*tdError*a;
	const float V_temp = DPG_V_alpha*fac*tdError;
	for (int i = 0; i < NUM_TILINGS; i++){
		DPG_T[TILE_FEATURES[i]] = DPG_T[TILE_FEATURES[i]] + T_temp;
		DPG_W[TILE_FEATURES[i]] = DPG_W[TILE_FEATURES[i]] + W_temp;
		DPG_V[TILE_FEATURES[i]] = DPG_V[TILE_FEATURES[i]] + V_temp;
	}
}

