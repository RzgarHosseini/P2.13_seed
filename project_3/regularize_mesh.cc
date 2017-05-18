/* ---------------------------------------------------------------------
 *
 * Copyright (C) 1999 - 2015 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------

 */



#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_reordering.h>

#include <deal.II/opencascade/utilities.h>
#include <deal.II/opencascade/boundary_lib.h>

#include <iostream>
#include <fstream>
#include <cmath>

using namespace dealii;


template<int dim, int spacedim>
void regularize_mesh (Triangulation<dim,spacedim> &tria,
                      const double limit_angle_fraction=.85)
{
    if(dim == 1)
        return; // Nothing to do

    bool has_cells_with_more_than_dim_faces_on_boundary = true;

    while(has_cells_with_more_than_dim_faces_on_boundary) {
        has_cells_with_more_than_dim_faces_on_boundary = false;

        for(auto cell: tria.active_cell_iterators()) {
            unsigned int boundary_face_counter = 0;
            for(unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
                if(cell->face(f)->at_boundary())
                    boundary_face_counter++;
            if(boundary_face_counter > dim) {
                has_cells_with_more_than_dim_faces_on_boundary = true;
                break;
            }
        }
        if(has_cells_with_more_than_dim_faces_on_boundary)
            tria.refine_global(1);
    }

    tria.refine_global(1);

    std::vector<bool> cells_to_remove(tria.n_active_cells(), false);
    std::vector<Point<spacedim> > vertices = tria.get_vertices();

    std::vector<bool> faces_to_remove(tria.n_raw_faces(),false);

    std::vector<CellData<dim> > cells_to_add;
    SubCellData                 subcelldata_to_add;
    std::vector<bool> vertices_touched(tria.n_vertices(), false);

    for(auto cell : tria.active_cell_iterators()) {
        double angle_fraction = 0;
        unsigned int vertex_at_corner = numbers::invalid_unsigned_int;

        if(dim == 2) {
            Tensor<1,spacedim> p0; p0[1] = 1;
            Tensor<1,spacedim> p1; p1[0] = 1;

            if(cell->face(0)->at_boundary() && cell->face(3)->at_boundary()) {
                p0 = cell->vertex(0) -  cell->vertex(2);
                p1 = cell->vertex(3) -  cell->vertex(2);
                vertex_at_corner = 2;
            } else  if(cell->face(3)->at_boundary() && cell->face(1)->at_boundary()) {
                p0 = cell->vertex(2) -  cell->vertex(3);
                p1 = cell->vertex(1) -  cell->vertex(3);
                vertex_at_corner = 3;
            } else  if(cell->face(1)->at_boundary() && cell->face(2)->at_boundary()) {
                p0 = cell->vertex(0) -  cell->vertex(1);
                p1 = cell->vertex(3) -  cell->vertex(1);
                vertex_at_corner = 1;
            } else  if(cell->face(2)->at_boundary() && cell->face(0)->at_boundary()) {
                p0 = cell->vertex(2) -  cell->vertex(0);
                p1 = cell->vertex(1) -  cell->vertex(0);
                vertex_at_corner = 0;
            }
            p0 /= p0.norm();
            p1 /= p1.norm();
            angle_fraction = std::acos(p0*p1)/numbers::PI;

        } else {
            Assert(false, ExcNotImplemented());
        }

        if(angle_fraction > limit_angle_fraction) {

           auto flags_removal = [&](unsigned int f1, unsigned int f2,
                                      unsigned int n1, unsigned int n2) -> void {
                cells_to_remove[cell->active_cell_index()] = true;
                cells_to_remove[cell->neighbor(n1)->active_cell_index()] = true;
                cells_to_remove[cell->neighbor(n2)->active_cell_index()] = true;

                faces_to_remove[cell->face(f1)->index()] = true;
                faces_to_remove[cell->face(f2)->index()] = true;

                faces_to_remove[cell->neighbor(n1)->face(f1)->index()] = true;
                faces_to_remove[cell->neighbor(n2)->face(f2)->index()] = true;
            };

            auto cell_creation = [&](
                        const unsigned int v0,
                        const unsigned int v1,
                        const unsigned int f0,
                        const unsigned int f1,

                        const unsigned int n0,
                        const unsigned int v0n0,
                        const unsigned int v1n0,

                        const unsigned int n1,
                        const unsigned int v0n1,
                        const unsigned int v1n1) {
                CellData<dim> c1, c2;
                CellData<1> l1, l2;

                c1.vertices[0] = cell->vertex_index(v0);
                c1.vertices[1] = cell->vertex_index(v1);
                c1.vertices[2] = cell->neighbor(n0)->vertex_index(v0n0);
                c1.vertices[3] = cell->neighbor(n0)->vertex_index(v1n0);

                c1.manifold_id = cell->manifold_id();
                c1.material_id = cell->material_id();

                c2.vertices[0] = cell->vertex_index(v0);
                c2.vertices[1] = cell->neighbor(n1)->vertex_index(v0n1);
                c2.vertices[2] = cell->vertex_index(v1);
                c2.vertices[3] = cell->neighbor(n1)->vertex_index(v1n1);

                c2.manifold_id = cell->manifold_id();
                c2.material_id = cell->material_id();

                l1.vertices[0] = cell->vertex_index(v0);
                l1.vertices[1] = cell->neighbor(n0)->vertex_index(v0n0);

                l1.boundary_id = cell->line(f0)->boundary_id();
                l1.manifold_id = cell->line(f0)->manifold_id();
                subcelldata_to_add.boundary_lines.push_back(l1);

                l2.vertices[0] = cell->vertex_index(v0);
                l2.vertices[1] = cell->neighbor(n1)->vertex_index(v0n1);

                l2.boundary_id = cell->line(f1)->boundary_id();
                l2.manifold_id = cell->line(f1)->manifold_id();
                subcelldata_to_add.boundary_lines.push_back(l2);

                cells_to_add.push_back(c1);
                cells_to_add.push_back(c2);
            };

            if(dim == 2) {
                switch(vertex_at_corner) {
                case 0:
                    flags_removal(0,2,3,1);
                    cell_creation(0,3, 0,2, 3,2,3,  1,1,3);
                    break;
                case 1:
                    flags_removal(1,2,3,0);
                    cell_creation(1,2, 2,1, 0,0,2, 3,3,2);
                    break;
                case 2:
                    flags_removal(3,0,1,2);
                    cell_creation(2,1, 3,0, 1,3,1, 2,0,1);
                    break;
                case 3:
                    flags_removal(3,1,0,2);
                    cell_creation(3,0, 1,3, 2,1,0, 0,2,0);
                    break;
                }
            } else {
                Assert(false, ExcNotImplemented());
            }
        }
    }

    // add the cells that were not marked as skipped
    for (auto cell : tria.active_cell_iterators())
    {
        if (cells_to_remove[cell->active_cell_index()] == false)
        {
            CellData<dim> c;
            for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_cell; ++v)
                c.vertices[v] = cell->vertex_index(v);
            c.manifold_id = cell->manifold_id();
            c.material_id = cell->material_id();
            cells_to_add.push_back(c);
        }
    }

    // Face counter for both dim == 2 and dim == 3
    typename Triangulation<dim,spacedim>::active_face_iterator
            face = tria.begin_active_face(),
            endf = tria.end_face();
    for (; face != endf; ++face)
        if (face->at_boundary() && faces_to_remove[face->index()] == false)
        {
            for (unsigned int l=0; l<GeometryInfo<dim>::lines_per_face; ++l)
            {
                CellData<1> line;
                if (dim == 2)
                {
                    for (unsigned int v=0; v<GeometryInfo<1>::vertices_per_cell; ++v)
                        line.vertices[v] = face->vertex_index(v);
                    line.boundary_id = face->boundary_id();
                    line.manifold_id = face->manifold_id();
                }
                else
                {
                    for (unsigned int v=0; v<GeometryInfo<1>::vertices_per_cell; ++v)
                        line.vertices[v] = face->line(l)->vertex_index(v);
                    line.boundary_id = face->line(l)->boundary_id();
                    line.manifold_id = face->line(l)->manifold_id();
                }
                subcelldata_to_add.boundary_lines.push_back(line);
            }
            if (dim == 3)
            {
                CellData<2> quad;
                for (unsigned int v=0; v<GeometryInfo<2>::vertices_per_cell; ++v)
                    quad.vertices[v] = face->vertex_index(v);
                quad.boundary_id = face->boundary_id();
                quad.manifold_id = face->manifold_id();
                subcelldata_to_add.boundary_quads.push_back(quad);
            }
        }
    GridTools::delete_unused_vertices(vertices, cells_to_add, subcelldata_to_add);
    GridReordering<dim,spacedim>::reorder_cells(cells_to_add, true);

    // Save manifolds
    auto manifold_ids = tria.get_manifold_ids();
    std::map<types::manifold_id, const Manifold<dim,spacedim>*> manifolds;
    // Set manifolds in new Triangulation
    for(auto manifold_id: manifold_ids)
        if(manifold_id != numbers::invalid_manifold_id)
            manifolds[manifold_id] = &tria.get_manifold(manifold_id);

    tria.clear();

    tria.create_triangulation(vertices, cells_to_add, subcelldata_to_add);

    // Restore manifolds
    for(auto manifold_id: manifold_ids)
        if(manifold_id != numbers::invalid_manifold_id)
            tria.set_manifold(manifold_id, *manifolds[manifold_id]);
}



int main ()
{
    const SphericalManifold<2> m0;

    Triangulation<2> tria1;
    GridGenerator::hyper_cube(tria1,-1,1);
    tria1.set_all_manifold_ids_on_boundary(0);
    tria1.set_manifold(0, m0);

    tria1.refine_global(2);
    regularize_mesh (tria1);
    //tria1.refine_global(1);

    Triangulation<2> tria2;
    GridGenerator::hyper_ball(tria2, Point<2>(), std::sqrt(2.0));
    tria2.set_all_manifold_ids_on_boundary(0);
    tria2.set_manifold(0, m0);
    tria2.refine_global(3);

    std::ofstream out ("grid-1.vtk");
    std::ofstream out2 ("grid-2.vtk");
    GridOut grid_out;
    grid_out.write_vtk (tria1, out);
    grid_out.write_vtk (tria2, out2);
}
