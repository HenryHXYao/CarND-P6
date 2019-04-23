/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 * Modified on: Apr 21, 2019, Hongxiang Yao
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
using std::normal_distribution;
using std::cout;
using std::endl;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 100;  // TODO: Set the number of particles
  //cout << "initialization" << endl;
  //cout << "x" << x << "/y" << y << "/theta" << theta << endl;  
  std::default_random_engine gen;
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);  
  
  for(int i = 0; i < num_particles; ++i){
    Particle particle;
    particle.id = i;
    particle.x = dist_x(gen);
    particle.y = dist_y(gen);  
    particle.theta = dist_theta(gen);
    particles.push_back(particle);
  }
  
  is_initialized = true;
  //for(int i = 0; i < num_particles; ++i){
    //cout << "particle_x" << particles[i].x << "/particle_y" << particles[i].y << "/particle_theta" << particles[i].theta << endl;  
  //}
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  
  std::default_random_engine gen;
  for (int i = 0;i < num_particles; ++i){
    double predicted_x, predicted_y;
    if (yaw_rate > 0.001) {
      predicted_x = particles[i].x + velocity/yaw_rate * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
      predicted_y = particles[i].y + velocity/yaw_rate * (-cos(particles[i].theta + yaw_rate * delta_t) + cos(particles[i].theta));
    }
    else {
      predicted_x = particles[i].x + velocity * delta_t * cos(particles[i].theta);
      predicted_y = particles[i].y + velocity * delta_t * sin(particles[i].theta);
    }
    double predicted_theta = particles[i].theta + yaw_rate * delta_t;
    //cout << "predicted_x" << predicted_x << "/predicted_y" << predicted_y << "/predicted_theta" << predicted_theta << endl; 
   
    normal_distribution<double> dist_x(predicted_x, std_pos[0]);
    normal_distribution<double> dist_y(predicted_y, std_pos[1]);
    normal_distribution<double> dist_theta(predicted_theta, std_pos[2]);  
    
    particles[i].x = dist_x(gen);
    particles[i].y = dist_y(gen);  
    particles[i].theta = dist_theta(gen);
    //cout << "particle_x" << particles[i].x << "/particle_y" << particles[i].y << "/particle_theta" << particles[i].theta << endl; 
  } 
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  double sum_weight = 0.0;
  for (int i = 0; i < num_particles; ++i){
    std::vector<int> associations;
    std::vector<double> sense_x;
    std::vector<double> sense_y;
    for (int j = 0; j < observations.size(); ++j ){
      // transform observations from vehicle coordinates to global coordinates    
      double sense_x_j = observations[j].x * cos(particles[i].theta) - observations[j].y * sin(particles[i].theta) + particles[i].x;
      double sense_y_j = observations[j].x * sin(particles[i].theta) + observations[j].y * cos(particles[i].theta) + particles[i].y;    
      // search among all landmarks within the sensor_range to find the association
      double min_dist = sensor_range;
      int id = -1;
      for (int k = 0; k < map_landmarks.landmark_list.size(); ++k){
        if (dist(particles[i].x, particles[i].y, map_landmarks.landmark_list[k].x_f, map_landmarks.landmark_list[k].y_f)<=                 sensor_range ){
          double current_dist = dist(map_landmarks.landmark_list[k].x_f, map_landmarks.landmark_list[k].y_f, sense_x_j, sense_y_j);
          if ( current_dist < min_dist){
            min_dist = current_dist;
            id = map_landmarks.landmark_list[k].id_i;
          }
        }
      }

      if (id != -1){
        associations.push_back(id);
        sense_x.push_back(sense_x_j);
        sense_y.push_back(sense_y_j);       
      }
    }
    
    // set associations for the current particle
    SetAssociations(particles[i], associations, sense_x, sense_y);
 
    // update the particle's weight
    particles[i].weight = 1;
    for (int k = 0; k < particles[i].associations.size(); ++k){
      particles[i].weight *= multiv_prob(std_landmark[0], std_landmark[1], sense_x[k], sense_y[k], map_landmarks.landmark_list[associations[k]-1].x_f, map_landmarks.landmark_list[associations[k]-1].y_f);    
    }
   sum_weight += particles[i].weight;
  }
  
   // weights normalization
  for (int i = 0; i < num_particles; ++i) {
    particles[i].weight /= sum_weight;
  }
}

void ParticleFilter::resample() {
  /**
   * Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
   std::vector<double> particles_weights;
   for (int i = 0; i < num_particles; ++i){
    particles_weights.push_back(particles[i].weight);
   }
   std::vector<Particle> new_particles;
   std::default_random_engine generator;
   std::discrete_distribution<int> distribution (particles_weights.cbegin(),particles_weights.cend());

  for (int i=0; i<num_particles; ++i) {
    int number = distribution(generator);
    new_particles.push_back(particles[number]);
  }
  particles = std::move(new_particles);
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
