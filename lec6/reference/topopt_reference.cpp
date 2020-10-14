#define TC_USE_DOUBLE
#define TC_IMAGE_IO
#include "taichi.h"
using namespace taichi;

int window_width = 1080, window_height = (window_width - 200) / 2;
using Vector = VectorND<2, real>;
using Vectori = VectorND<2, int>;
template <typename T>
using Array = ArrayND<2, T>;

// Array is like a 2D C++ array. The element can be either real or 2D Vector
// In this file, we use real = double

class TopologyOptimization {
 public:
  Vectori cell_res, node_res;
  real volume_fraction, minimum_density, penalty;
  real iteration_time, start_time, total_time, objective;
  bool running;
  int iteration;
  int bc_type;
  real filter_radius;
  real change_limit;
  Array<Vector> f, u, last_u;
  Array<real> density;
  CPUCGHexFEMSolver<2> fem;
  std::mutex mut;

  typename HexFEMSolver<2>::BoundaryCondition boundary_condition;

  void initialize_boundary_condition() {
    f.reset(Vector2(0));
    boundary_condition.clear();
    switch (bc_type) {
      case 1:  // MBB Beam
        // Load
        f[0][cell_res[1]][1] = -1;
        // Dirichlet BCs (fixed degrees of freedom)
        // Boundary condition = {(node_x, node_y), axis (0 for x, 1 for y) to
        // fix, offset (usually zero)}
        for (int j = 0; j < node_res[1]; j++)
          boundary_condition.push_back({Vector2i(0, j), 0, 0});
        boundary_condition.push_back({Vector2i(cell_res[0], 0), 1, 0});
        break;
      case 2:  // Cantilever Beam (TODO, Task 3)
        boundary_condition.push_back({Vector2i(0, 0), 0, 0});
        boundary_condition.push_back({Vector2i(0, 0), 1, 0});

        boundary_condition.push_back({Vector2i(0, cell_res[1]), 0, 0});
        boundary_condition.push_back({Vector2i(0, cell_res[1]), 1, 0});

        f[cell_res[0]][cell_res[1] / 2][1] = -1;
        break;
      case 3:  // Cantilever Beam (TODO, Task 3)
        boundary_condition.push_back({Vector2i(0, 0), 0, 0});
        boundary_condition.push_back({Vector2i(0, 0), 1, 0});

        boundary_condition.push_back({Vector2i(cell_res[0], 0), 0, 0});
        boundary_condition.push_back({Vector2i(cell_res[0], 0), 1, 0});

        for (int i = 0; i < node_res[0]; i++)
          f[i][cell_res[1]][1] = -1;

        break;
    }
  }

  TopologyOptimization() {
    running = false;
    std::thread th([this]() {
      // Main loop in a separate thread
      while (1) {
        if (running) {
          auto change = iterate();
          if (change < 0.005)
            running = false;
        }
        taichi::Time::sleep(0.01);
      }
    });
    th.detach();
  }

  void initialize() {
    start_time = taichi::Time::get_time();
    iteration_time = total_time = iteration = objective = 0;
    minimum_density = 1e-2_f;
    node_res = cell_res + Vector2i(1, 1);
    fem.initialize(node_res, penalty);
    u.initialize(node_res, Vector(0.0f));
    last_u = u;
    density.initialize(cell_res, volume_fraction);
    f.initialize(node_res, Vector(0.0f));
    initialize_boundary_condition();
    fem.set_boundary_condition(boundary_condition);
    running = true;
  }

  // Task 1:
  Array<real> optimality_criteria(const Array<real> &s) {
    Array<real> new_density = density.same_shape();
    // TODO: implement the Optimality Criteria algorithm
    // Note: You should make use of
    //    minimum_density, change_limit, volume_fraction, cell_res, density here
    real lower = 0, upper = 1e15;
    while (lower * (1 + 1e-15_f) < upper) {
      real mid = 0.5_f * (lower + upper);
      for (auto &ind : new_density.get_region()) {
        real old = density[ind];
        new_density[ind] = clamp(old * std::sqrt(s[ind] / mid),
                                 old - change_limit, old + change_limit);
        new_density[ind] = clamp(new_density[ind], minimum_density, 1.0_f);
      }
      if (new_density.sum() - volume_fraction * cell_res.prod() > 0)
        lower = mid;
      else
        upper = mid;
    }
    return new_density;
  }

  Array<real> sensitivity_filtering(const Array<real> &s) const {
    Array<real> s_filtered = s.same_shape();
    if (filter_radius == 0.0) {
      s_filtered = s;  // no filtering. Directly copy.
    } else {
      for (int i = 0; i < cell_res[0]; i++) {
        for (int j = 0; j < cell_res[1]; j++) {
          // Task 2: Sensitivity filtering
          // TODO: replace the follow line with filtering.
          // s_filtered[i][j] = s[i][j];
          // NOTE: density and s have size of ``cell_res''.
          //       Be careful not to access the undefined region outside the
          //       range.
          real total_s = 0, total_w = 0;
          int radius_int = std::ceil(filter_radius);
          for (int dx = -radius_int; dx <= radius_int; dx++) {
            for (int dy = -radius_int; dy <= radius_int; dy++) {
              int ni = i + dx;
              int nj = j + dy;
              if (ni < 0 || nj < 0 || ni >= cell_res[0] || nj >= cell_res[1]) {
                continue;
              }
              real nu = density[ni][nj];
              real w = std::max(0.0, filter_radius - std::hypot(dx, dy));
              total_s += w * nu * s[ni][nj];
              total_w += w * nu;
            }
          }
          s_filtered[i][j] = total_s / total_w;
        }
      }
    }
    return s_filtered;
  }

  // Optimization iteration.
  // returns: max change
  real iterate() {
    std::lock_guard<std::mutex> _(mut);
    real start_t = taichi::Time::get_time();
    Array<real> s = density.same_shape(0.0f);  // sensitivity
    auto u = fem.solve(density, f, last_u, s, objective);
    last_u = u;
    s = sensitivity_filtering(s);
    auto new_density = optimality_criteria(s);
    auto diff = new_density - density;
    real change = diff.abs_max();
    density = new_density;
    iteration += 1;
    iteration_time = 1000 * (taichi::Time::get_time() - start_t);
    total_time = taichi::Time::get_time() - start_time;
    return change;
  }

  void render(Canvas &canvas) {
    real scale_x = density.get_res()[0] / real(window_width - 200);
    real scale_y = density.get_res()[1] / real(window_height);
    for (auto &ind : Region2D(Vector2i(0, 0),
                              Vector2i(window_width - 200, window_height))) {
      Vectori coord(int32(ind.get_ipos()[0] * scale_x),
                    int32((ind.get_ipos()[1]) * scale_y));
      if (density.inside(coord))
        canvas.img[ind] = Vector4(1 - density[coord]);
    }
  }
};

TopologyOptimization opt;

int main() {
  GUI gui("Topology Optimization", Vector2i(window_width, window_height),
          false);
  int res = 50, penalty = 3, bc_type = 1;
  real volume_fraction = 0.5, filter_radius = 1.5, change_limit = 0.2;
  auto start = [&]() {
    opt.running = false;
    std::lock_guard<std::mutex> _(opt.mut);
    opt.cell_res = Vector2i(res, res / 2);
    opt.volume_fraction = volume_fraction;
    opt.penalty = penalty;
    opt.filter_radius = filter_radius;
    opt.change_limit = change_limit;
    opt.bc_type = bc_type;
    opt.initialize();
  };
  start();
  gui.button("(Re)start", start)
      .slider("Resolution", res, 10, 200)
      .slider("BC Type", bc_type, 1, 3)
      .slider("Volume Fraction (c)", volume_fraction, 0.1, 0.99)
      .slider("OC Change Limit (M)", change_limit, 0.01, 1.0)
      .slider("SIMP Penalty", penalty, 1, 4)
      .slider("Filter Radius", filter_radius, 0.0, 5.0)
      .label("Iteration", opt.iteration)
      .label("Iter. Time (ms)", opt.iteration_time)
      .label("Total Time (s)", opt.total_time)
      .label("Objective", opt.objective)
      .button("Save Image", [&] { gui.screenshot(); });
  while (1) {
    gui.canvas->clear(Vector4(1));
    opt.render(gui.get_canvas());
    gui.update();
  }
  return 0;
};
