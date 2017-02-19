#include "dpg.hpp"

const int N_EPISODES	= 10000;
const int MAX_STEPS 	= 5000;

int simulateEpisode(const int, float);
int simulateEpisodeTest(const int);

int main(int argc, char* argv[]){
	for (int i = 0; i < MEMORY_SIZE; i++){
		DPG_W[i] = float(rand())/RAND_MAX*0.01;
		DPG_T[i] = float(rand())/RAND_MAX*0.01;//0.0;
		DPG_V[i] = float(rand())/RAND_MAX*0.01;//0.0;
	}

	float fac = 1.0;
	for (int i = 0; i < N_EPISODES; i++){
		if (i % 5 == 0)
			fac = fac*0.95;

		srand(i);
		const int rTrain = simulateEpisode(MAX_STEPS, fac);

		if( i % 100 == 0){
			const int rTest = simulateEpisodeTest(MAX_STEPS);
			printf("Training Episode: %d; Train: %d; Finished Test In: %d\n", i+1, rTrain, rTest);
		}
	}
}

int simulateEpisode(const int steps, float fac){
	MCarInit();

	int cStep = 0; 
	while(true){
		// Get the current state-action pair
		float s_t[] = {mcar_position, mcar_velocity};
		buildTileFeatures(2, &s_t[0]);
		float a_t = pickActionRandomTiles(1.0);

		//std::cout<<a_t<<std::endl;		
		/*s
		for (int i = 0; i < NUM_TILINGS; i++){
			std::cout<<"("<<DPG_T[TILE_FEATURES[i]]<<","<<TILE_FEATURES[i]<<")"<<",";
		}
		std::cout<<std::endl;
		//*/
		//std::cout<<a_t<<std::endl;
		
		// Execute
		MCarStep(a_t);

		// Get next state and action pair
		float s_t_1[] = {mcar_position, mcar_velocity};
		buildTileFeaturesNext(2, &s_t_1[0]);
		float a_t_1 = pickActionTilesNext();

		// Check if mountain car is at goal
		bool NEXT_FLAG = false;
		float r_t_1 = -1.0;//(mcar_goal_position - mcar_position);
		//std::cout<<r_t_1<<std::endl;
		if (MCarAtGoal()){
			NEXT_FLAG = true;
			r_t_1 = 0.0;
		}

		// Update the weights
		DPGUpdatesTiles(a_t, a_t_1, r_t_1, NEXT_FLAG, fac);
		if ((++cStep) >= steps || NEXT_FLAG){
			break;
		}
	}
	return cStep;
}

int simulateEpisodeTest(const int steps){
	MCarInit();

	int cStep = 0; 
	while(true){
		// Get the current state-action pair
		float s_t[] = {mcar_position, mcar_velocity};
		buildTileFeatures(2, &s_t[0]);

		float a_t = pickActionTiles();
		//std::cout<<a_t_1<<",";

		//std::cout<<a_t<<std::endl;
		// Execute
		MCarStep(a_t);

		if (++cStep >= steps || MCarAtGoal())
			break;
	}
	return cStep;
}