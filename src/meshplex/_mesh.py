from __future__ import annotations
import math
import pathlib
import warnings
# import meshio
import npx
import numpy as np
from ._exceptions import MeshplexError
from ._helpers import _dot, _multiply, grp_start_len
__all__ = ['Mesh']


class Mesh:

    def __init__(self, points, cells, sort_cells=False):
        points = np.asarray(points)
        cells = np.asarray(cells)
        self.edges = None
        if sort_cells:
            cells = np.sort(cells, axis=1)
            cells = cells[cells[:, 0].argsort()]
        assert len(cells.shape) == 2, f'Illegal cells shape {cells.shape}'
        self.n = cells.shape[1]
        self._points = np.asarray(points)
        self._points.setflags(write=False)
        self.idx = [np.asarray(cells).T]
        for _ in range(1, self.n - 1):
            m = len(self.idx[-1])
            r = np.arange(m)
            k = np.array([np.roll(r, -i) for i in range(1, m)])
            self.idx.append(self.idx[-1][k])
        self._is_point_used = None
        self._is_boundary_facet = None
        self._is_boundary_facet_local = None
        self.facets = None
        self._boundary_facets = None
        self._interior_facets = None
        self._is_interior_point = None
        self._is_boundary_point = None
        self._is_boundary_cell = None
        self._cells_facets = None
        self.subdomains = {}
        self._reset_point_data()

    def _reset_point_data(self):
        """Reset all data that changes when point coordinates changes."""
        self._half_edge_coords = None
        self._ei_dot_ei = None
        self._cell_centroids = None
        self._volumes = None
        self._integral_x = None
        self._signed_cell_volumes = None
        self._circumcenters = None
        self._cell_circumradii = None
        self._cell_heights = None
        self._ce_ratios = None
        self._cell_partitions = None
        self._control_volumes = None
        self._signed_circumcenter_distances = None
        self._circumcenter_facet_distances = None
        self._cv_centroids = None
        self._cvc_cell_mask = None
        self._cv_cell_mask = None

    def __repr__(self):
        name = {(2): 'line', (3): 'triangle', (4): 'tetra'}[self.cells(
            'points').shape[1]]
        num_points = len(self.points)
        num_cells = len(self.cells('points'))
        string = (
            f'<meshplex {name} mesh, {num_points} points, {num_cells} cells>')
        return string

    @property
    def points(self):
        return self._points

    @points.setter
    def points(self, new_points):
        new_points = np.asarray(new_points)
        assert new_points.shape == self._points.shape
        self._points = new_points
        self._reset_point_data()

    def set_points(self, new_points, idx=slice(None)):
        self.points.setflags(write=True)
        self.points[idx] = new_points
        self.points.setflags(write=False)
        self._reset_point_data()

    def cells(self, which):
        if which == 'points':
            return self.idx[0].T
        elif which == 'facets':
            assert self._cells_facets is not None
            return self._cells_facets
        assert which == 'edges'
        assert self.n == 3
        assert self._cells_facets is not None
        return self._cells_facets

    @property
    def half_edge_coords(self):
        if self._half_edge_coords is None:
            self._compute_cell_values()
        assert self._half_edge_coords is not None
        return self._half_edge_coords

    @property
    def ei_dot_ei(self):
        if self._ei_dot_ei is None:
            self._compute_cell_values()
        assert self._ei_dot_ei is not None
        return self._ei_dot_ei

    @property
    def cell_heights(self):
        if self._cell_heights is None:
            self._compute_cell_values()
        assert self._cell_heights is not None
        return self._cell_heights

    @property
    def edge_lengths(self):
        if self._volumes is None:
            self._compute_cell_values()
        assert self._volumes is not None
        return self._volumes[0]

    @property
    def facet_areas(self):
        if self.n == 2:
            assert self.facets is not None
            return np.ones(len(self.facets['points']))
        if self._volumes is None:
            self._compute_cell_values()
        assert self._volumes is not None
        return self._volumes[-2]

    @property
    def cell_volumes(self):
        if self._volumes is None:
            self._compute_cell_values()
        assert self._volumes is not None
        return self._volumes[-1]

    @property
    def cell_circumcenters(self):
        """Get the center of the circumsphere of each cell."""
        if self._circumcenters is None:
            self._compute_cell_values()
        assert self._circumcenters is not None
        return self._circumcenters[-1]

    @property
    def cell_circumradius(self):
        """Get the circumradii of all cells"""
        if self._cell_circumradii is None:
            self._compute_cell_values()
        assert self._cell_circumradii is not None
        return self._cell_circumradii

    @property
    def cell_partitions(self):
        """Each simplex can be subdivided into parts that a closest to each corner.

        This method gives those parts, like ce_ratios associated with each edge.

        """
        if self._cell_partitions is None:
            self._compute_cell_values()
        assert self._cell_partitions is not None
        return self._cell_partitions

    @property
    def circumcenter_facet_distances(self):
        if self._circumcenter_facet_distances is None:
            self._compute_cell_values()
        assert self._circumcenter_facet_distances is not None
        return self._circumcenter_facet_distances

    def get_control_volume_centroids(self, cell_mask=None):
        """The centroid of any volume V is given by



        .. math::

          c = \\int_V x / \\int_V 1.



        The denominator is the control volume. The numerator can be computed by making

        use of the fact that the control volume around any vertex is composed of right

        triangles, two for each adjacent cell.



        Optionally disregard the contributions from particular cells. This is useful,

        for example, for temporarily disregarding flat cells on the boundary when

        performing Lloyd mesh optimization.

        """
        if self._cv_centroids is None or np.any(cell_mask != self.
            _cvc_cell_mask):
            if self._integral_x is None:
                self._compute_cell_values()
            if cell_mask is None:
                idx = Ellipsis
            else:
                cell_mask = np.asarray(cell_mask)
                assert cell_mask.dtype == bool
                assert cell_mask.shape == (self.idx[-1].shape[-1],)
                idx = tuple((self.n - 1) * [slice(None)] + [~cell_mask])
            integral_p = npx.sum_at(self._integral_x[idx], self.idx[-1][idx
                ], len(self.points))
            cv = self.get_control_volumes(cell_mask)
            self._cv_centroids = (integral_p.T / cv).T
            self._cvc_cell_mask = cell_mask
        return self._cv_centroids

    @property
    def control_volume_centroids(self):
        return self.get_control_volume_centroids()

    @property
    def ce_ratios(self):
        """The covolume-edgelength ratios."""
        if self._ce_ratios is None:
            self._ce_ratios = self.cell_partitions[0] / self.ei_dot_ei * 2 * (
                self.n - 1)
        return self._ce_ratios

    @property
    def signed_circumcenter_distances(self):
        if self._signed_circumcenter_distances is None:
            if self._cells_facets is None:
                self.create_facets()
            self._signed_circumcenter_distances = npx.sum_at(self.
                circumcenter_facet_distances.T, self.cells('facets'), self.
                facets['points'].shape[0])[self.is_interior_facet]
        return self._signed_circumcenter_distances

    def _compute_cell_values(self, mask=None):
        """Computes the volumes of all edges, facets, cells etc. in the mesh. It starts

        off by computing the (squared) edge lengths, then complements the edge with one

        vertex to form face. It computes an orthogonal basis of the face (with modified

        Gram-Schmidt), and from that gets the height of all faces. From this, the area

        of the face is computed. Then, it complements again to form the 3-simplex,

        again forms an orthogonal basis with Gram-Schmidt, and so on.

        """
        if mask is None:
            mask = slice(None)
        e = self.points[self.idx[-1][..., mask]]
        e0 = e[0]
        diff = e[1] - e[0]
        orthogonal_basis = np.array([diff])
        volumes2 = [_dot(diff, self.n - 1)]
        circumcenters = [0.5 * (e[0] + e[1])]
        vv = _dot(diff, self.n - 1)
        circumradii2 = 0.25 * vv
        sqrt_vv = np.sqrt(vv)
        lmbda = 0.5 * sqrt_vv
        sumx = np.array(e + circumcenters[-1])
        partitions = 0.5 * np.array([sqrt_vv, sqrt_vv])
        norms2 = np.array(volumes2)
        for kk, idx in enumerate(self.idx[:-1][::-1]):
            p0 = self.points[idx][:, mask]
            v = p0 - e0
            for w, w_dot_w in zip(orthogonal_basis, norms2):
                w_dot_v = np.einsum('...k,...k->...', w, v)
                alpha = np.divide(w_dot_v, w_dot_w, where=w_dot_w > 0.0,
                    out=w_dot_v)
                v -= _multiply(w, alpha, self.n - 1 - kk)
            vv = np.einsum('...k,...k->...', v, v)
            k0 = 0
            e0 = e0[k0]
            orthogonal_basis = np.row_stack([orthogonal_basis[:, k0], [v[k0]]])
            norms2 = np.row_stack([norms2[:, k0], [vv[k0]]])
            volumes2.append(volumes2[-1][0] * vv[k0] / (kk + 2) ** 2)
            c = circumcenters[-1]
            p0c2 = _dot(p0 - c, self.n - 1 - kk)
            a = 0.5 * (p0c2 - circumradii2)
            sqrt_vv = np.sqrt(vv)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                lmbda = a / sqrt_vv
                sigma_k0 = a[k0] / vv[k0]
            lmbda2_k0 = sigma_k0 * a[k0]
            circumradii2 = lmbda2_k0 + circumradii2[k0]
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                circumcenters.append(c[k0] + _multiply(v[k0], sigma_k0, 
                    self.n - 2 - kk))
            sumx += circumcenters[-1]
            partitions *= lmbda / (kk + 2)
        integral_x = _multiply(sumx, partitions / self.n, self.n)
        if np.all(mask == slice(None)):
            self._ei_dot_ei = volumes2[0]
            self._half_edge_coords = diff
            self._volumes = [np.sqrt(v2) for v2 in volumes2]
            self._circumcenter_facet_distances = lmbda
            self._cell_heights = sqrt_vv
            self._cell_circumradii = np.sqrt(circumradii2)
            self._circumcenters = circumcenters
            self._cell_partitions = partitions
            self._integral_x = integral_x
        else:
            assert self._ei_dot_ei is not None
            self._ei_dot_ei[:, mask] = volumes2[0]
            assert self._half_edge_coords is not None
            self._half_edge_coords[:, mask] = diff
            assert self._volumes is not None
            for k in range(len(self._volumes)):
                self._volumes[k][..., mask] = np.sqrt(volumes2[k])
            assert self._circumcenter_facet_distances is not None
            self._circumcenter_facet_distances[..., mask] = lmbda
            assert self._cell_heights is not None
            self._cell_heights[..., mask] = sqrt_vv
            assert self._cell_circumradii is not None
            self._cell_circumradii[mask] = np.sqrt(circumradii2)
            assert self._circumcenters is not None
            for k in range(len(self._circumcenters)):
                self._circumcenters[k][..., mask, :] = circumcenters[k]
            assert self._cell_partitions is not None
            self._cell_partitions[..., mask] = partitions
            assert self._integral_x is not None
            self._integral_x[..., mask, :] = integral_x

    @property
    def signed_cell_volumes(self):
        """Signed volumes of an n-simplex in nD."""
        if self._signed_cell_volumes is None:
            self._signed_cell_volumes = self.compute_signed_cell_volumes()
        return self._signed_cell_volumes

    def compute_signed_cell_volumes(self, idx=slice(None)):
        """Signed volume of a simplex in nD. Note that signing only makes sense for

        n-simplices in R^n.

        """
        n = self.points.shape[1]
        assert self.n == self.points.shape[1
            ] + 1, f'Signed areas only make sense for n-simplices in in nD. Got {n}D points.'
        if self.n == 3:
            x = self.half_edge_coords
            assert x is not None
            return (x[0, idx, 1] * x[2, idx, 0] - x[0, idx, 0] * x[2, idx, 1]
                ) / 2
        cp = self.points[self.cells('points')]
        cp1 = np.concatenate([cp, np.ones(cp.shape[:-1] + (1,))], axis=-1)
        sign = -1 if n % 2 == 1 else 1
        return sign * np.linalg.det(cp1) / math.factorial(n)

    def compute_cell_centroids(self, idx=slice(None)):
        return np.sum(self.points[self.cells('points')[idx]], axis=1) / self.n

    @property
    def cell_centroids(self):
        """The centroids (barycenters, midpoints of the circumcircles) of all

        simplices."""
        if self._cell_centroids is None:
            self._cell_centroids = self.compute_cell_centroids()
        return self._cell_centroids
    cell_barycenters = cell_centroids

    @property
    def cell_incenters(self):
        """Get the midpoints of the inspheres."""
        abc = self.facet_areas / np.sum(self.facet_areas, axis=0)
        return np.einsum('ij,jik->jk', abc, self.points[self.cells('points')])

    @property
    def cell_inradius(self):
        """Get the inradii of all cells"""
        return (self.n - 1) * self.cell_volumes / np.sum(self.facet_areas,
            axis=0)

    @property
    def is_point_used(self):
        if self._is_point_used is None:
            self._is_point_used = np.zeros(len(self.points), dtype=bool)
            self._is_point_used[self.cells('points')] = True
        return self._is_point_used

    def write(self, filename, point_data=None, cell_data=None, field_data=None
        ):
        if self.points.shape[1] == 2:
            n = len(self.points)
            a = np.ascontiguousarray(np.column_stack([self.points, np.zeros
                (n)]))
        else:
            a = self.points
        if self.cells('points').shape[1] == 3:
            cell_type = 'triangle'
        else:
            assert self.cells('points').shape[1
                ] == 4, 'Only triangles/tetrahedra supported'
            cell_type = 'tetra'
        import meshio
        meshio.Mesh(a, {cell_type: self.cells('points')}, point_data=
            point_data, cell_data=cell_data, field_data=field_data).write(
            filename)

    def get_vertex_mask(self, subdomain=None):
        if subdomain is None:
            return slice(None)
        if subdomain not in self.subdomains:
            self._mark_vertices(subdomain)
        return self.subdomains[subdomain]['vertices']

    def get_edge_mask(self, subdomain=None):
        """Get faces which are fully in subdomain."""
        if subdomain is None:
            return slice(None)
        if subdomain not in self.subdomains:
            self._mark_vertices(subdomain)
        is_in = self.subdomains[subdomain]['vertices'][self.idx[-1]]
        is_inside = np.all(is_in, axis=tuple(range(1)))
        if subdomain.is_boundary_only:
            is_inside = is_inside & self.is_boundary_facet
        return is_inside

    def get_face_mask(self, subdomain):
        """Get faces which are fully in subdomain."""
        if subdomain is None:
            return slice(None)
        if subdomain not in self.subdomains:
            self._mark_vertices(subdomain)
        is_in = self.subdomains[subdomain]['vertices'][self.idx[-1]]
        n = len(is_in.shape)
        is_inside = np.all(is_in, axis=tuple(range(n - 2)))
        if subdomain.is_boundary_only:
            is_inside = is_inside & self.is_boundary_facet_local
        return is_inside

    def get_cell_mask(self, subdomain=None):
        if subdomain is None:
            return slice(None)
        if subdomain.is_boundary_only:
            return np.array([])
        if subdomain not in self.subdomains:
            self._mark_vertices(subdomain)
        is_in = self.subdomains[subdomain]['vertices'][self.idx[-1]]
        n = len(is_in.shape)
        return np.all(is_in, axis=tuple(range(n - 1)))

    def _mark_vertices(self, subdomain):
        """Mark faces/edges which are fully in subdomain."""
        if subdomain is None:
            is_inside = np.ones(len(self.points), dtype=bool)
        else:
            is_inside = subdomain.is_inside(self.points.T).T
            if subdomain.is_boundary_only:
                is_inside &= self.is_boundary_point
        self.subdomains[subdomain] = {'vertices': is_inside}

    def create_facets(self):
        """Set up facet->point and facet->cell relations."""
        if self.n == 2:
            idx = self.idx[0].flatten()
        else:
            idx = self.idx[1]
            idx = idx.reshape(idx.shape[0], -1)
        idx = np.sort(idx, axis=0).T
        a_unique, inv, cts = npx.unique_rows(idx, return_inverse=True,
            return_counts=True)
        if np.any(cts > 2):
            num_weird_edges = np.sum(cts > 2)
            msg = (
                f'Found {num_weird_edges} facets with more than two neighboring cells. Something is not right.'
                )
            _, inv, cts = npx.unique_rows(np.sort(self.cells('points')),
                return_inverse=True, return_counts=True)
            if np.any(cts > 1):
                msg += ' The following cells are equal:\n'
                for multiple_idx in np.where(cts > 1)[0]:
                    msg += str(np.where(inv == multiple_idx)[0])
            raise MeshplexError(msg)
        self._is_boundary_facet_local = (cts[inv] == 1).reshape(self.idx[0]
            .shape)
        self._is_boundary_facet = cts == 1
        self.facets = {'points': a_unique}
        self._cells_facets = inv.reshape(self.n, -1).T
        if self.n == 3:
            self.edges = self.facets
            self._facets_cells = None
            self._facets_cells_idx = None
        elif self.n == 4:
            self.faces = self.facets

    @property
    def is_boundary_facet_local(self):
        if self._is_boundary_facet_local is None:
            self.create_facets()
        assert self._is_boundary_facet_local is not None
        return self._is_boundary_facet_local

    @property
    def is_boundary_facet(self):
        if self._is_boundary_facet is None:
            self.create_facets()
        assert self._is_boundary_facet is not None
        return self._is_boundary_facet

    @property
    def is_interior_facet(self):
        return ~self.is_boundary_facet

    @property
    def is_boundary_cell(self):
        if self._is_boundary_cell is None:
            assert self.is_boundary_facet_local is not None
            self._is_boundary_cell = np.any(self.is_boundary_facet_local,
                axis=0)
        return self._is_boundary_cell

    @property
    def boundary_facets(self):
        if self._boundary_facets is None:
            self._boundary_facets = np.where(self.is_boundary_facet)[0]
        return self._boundary_facets

    @property
    def interior_facets(self):
        if self._interior_facets is None:
            self._interior_facets = np.where(~self.is_boundary_facet)[0]
        return self._interior_facets

    @property
    def is_boundary_point(self):
        if self._is_boundary_point is None:
            self._is_boundary_point = np.zeros(len(self.points), dtype=bool)
            i = 0 if self.n == 2 else 1
            self._is_boundary_point[self.idx[i][..., self.
                is_boundary_facet_local]] = True
        return self._is_boundary_point

    @property
    def is_interior_point(self):
        if self._is_interior_point is None:
            self._is_interior_point = (self.is_point_used & ~self.
                is_boundary_point)
        return self._is_interior_point

    @property
    def facets_cells(self):
        if self._facets_cells is None:
            self._compute_facets_cells()
        return self._facets_cells

    def _compute_facets_cells(self):
        """This creates edge->cells relations. While it's not necessary for many

        applications, it sometimes does come in handy, for example for mesh

        manipulation.

        """
        if self.facets is None:
            self.create_facets()
        edges_flat = self.cells('edges').flat
        idx_sort = np.argsort(edges_flat)
        sorted_edges = edges_flat[idx_sort]
        idx_start, count = grp_start_len(sorted_edges)
        assert np.all((count == 1) == self.is_boundary_facet)
        assert np.all((count == 2) == self.is_interior_facet)
        idx_start_count_1 = idx_start[self.is_boundary_facet]
        idx_start_count_2 = idx_start[self.is_interior_facet]
        res1 = idx_sort[idx_start_count_1]
        res2 = idx_sort[np.array([idx_start_count_2, idx_start_count_2 + 1])]
        edge_id_boundary = sorted_edges[idx_start_count_1]
        edge_id_interior = sorted_edges[idx_start_count_2]
        self._facets_cells = {'boundary': np.array([edge_id_boundary, res1 //
            3, res1 % 3]), 'interior': np.array([edge_id_interior, *(res2 //
            3), *(res2 % 3)])}
        self._facets_cells_idx = None

    @property
    def facets_cells_idx(self):
        if self._facets_cells_idx is None:
            if self._facets_cells is None:
                self._compute_facets_cells()
            assert self.is_boundary_facet is not None
            num_edges = len(self.facets['points'])
            self._facets_cells_idx = np.empty(num_edges, dtype=int)
            num_b = np.sum(self.is_boundary_facet)
            num_i = np.sum(self.is_interior_facet)
            self._facets_cells_idx[self.facets_cells['boundary'][0]
                ] = np.arange(num_b)
            self._facets_cells_idx[self.facets_cells['interior'][0]
                ] = np.arange(num_i)
        return self._facets_cells_idx

    def remove_dangling_points(self):
        """Remove all points which aren't part of an array"""
        is_part_of_cell = np.zeros(self.points.shape[0], dtype=bool)
        is_part_of_cell[self.cells('points').flat] = True
        new_point_idx = np.cumsum(is_part_of_cell) - 1
        self._points = self._points[is_part_of_cell]
        for k in range(len(self.idx)):
            self.idx[k] = new_point_idx[self.idx[k]]
        if self._control_volumes is not None:
            self._control_volumes = self._control_volumes[is_part_of_cell]
        if self._cv_centroids is not None:
            self._cv_centroids = self._cv_centroids[is_part_of_cell]
        if self.facets is not None:
            self.facets['points'] = new_point_idx[self.facets['points']]
        if self._is_interior_point is not None:
            self._is_interior_point = self._is_interior_point[is_part_of_cell]
        if self._is_boundary_point is not None:
            self._is_boundary_point = self._is_boundary_point[is_part_of_cell]
        if self._is_point_used is not None:
            self._is_point_used = self._is_point_used[is_part_of_cell]
        return np.sum(~is_part_of_cell)

    @property
    def q_radius_ratio(self):
        """Ratio of incircle and circumcircle ratios times (n-1). ("Normalized shape

        ratio".) Is 1 for the equilateral simplex, and is often used a quality measure

        for the cell.

        """
        if self.n == 3:
            a, b, c = self.edge_lengths
            return (-a + b + c) * (a - b + c) * (a + b - c) / (a * b * c)
        return (self.n - 1) * self.cell_inradius / self.cell_circumradius

    def remove_cells(self, remove_array):
        """Remove cells and take care of all the dependent data structures. The input

        argument `remove_array` can be a boolean array or a list of indices.

        """
        remove_array = np.asarray(remove_array)
        if len(remove_array) == 0:
            return 0
        if remove_array.dtype == int:
            keep = np.ones(len(self.cells('points')), dtype=bool)
            keep[remove_array] = False
        else:
            assert remove_array.dtype == bool
            keep = ~remove_array
        assert len(keep) == len(self.cells('points')
            ), 'Wrong length of index array.'
        if np.all(keep):
            return 0
        if self._cells_facets is not None:
            if self._facets_cells is None:
                self._compute_facets_cells()
            facet_ids = self.cells('facets')[~keep].flatten()
            facet_ids = facet_ids[self.is_interior_facet[facet_ids]]
            idx = self.facets_cells_idx[facet_ids]
            cell_id = self.facets_cells['interior'][1:3, idx].T
            local_facet_id = self.facets_cells['interior'][3:5, idx].T
            self._is_boundary_facet_local[local_facet_id, cell_id] = True
            self._is_boundary_facet_local = self._is_boundary_facet_local[:,
                keep]
            if self._is_boundary_cell is not None:
                self._is_boundary_cell[cell_id] = True
                self._is_boundary_cell = self._is_boundary_cell[keep]
            keep_b_ec = keep[self.facets_cells['boundary'][1]]
            keep_i_ec0, keep_i_ec1 = keep[self.facets_cells['interior'][1:3]]
            keep_i_0 = keep_i_ec0 & ~keep_i_ec1
            keep_i_1 = keep_i_ec1 & ~keep_i_ec0
            self._facets_cells['boundary'] = np.array([np.concatenate([self
                ._facets_cells['boundary'][0, keep_b_ec], self.
                _facets_cells['interior'][0, keep_i_0], self._facets_cells[
                'interior'][0, keep_i_1]]), np.concatenate([self.
                _facets_cells['boundary'][1, keep_b_ec], self._facets_cells
                ['interior'][1, keep_i_0], self._facets_cells['interior'][2,
                keep_i_1]]), np.concatenate([self._facets_cells['boundary']
                [2, keep_b_ec], self._facets_cells['interior'][3, keep_i_0],
                self._facets_cells['interior'][4, keep_i_1]])])
            keep_i = keep_i_ec0 & keep_i_ec1
            self._facets_cells['interior'] = self._facets_cells['interior'][
                :, keep_i]
            num_facets_old = len(self.facets['points'])
            adjacent_facets, counts = np.unique(self.cells('facets')[~keep]
                .flat, return_counts=True)
            is_facet_removed = (counts == 2) | (counts == 1
                ) & self._is_boundary_facet[adjacent_facets]
            self._is_boundary_facet[adjacent_facets[~is_facet_removed]] = True
            assert self._is_boundary_facet is not None
            keep_facets = np.ones(len(self._is_boundary_facet), dtype=bool)
            keep_facets[adjacent_facets[is_facet_removed]] = False
            assert self.facets is not None
            assert len(self.facets) == 1
            self.facets['points'] = self.facets['points'][keep_facets]
            self._is_boundary_facet = self._is_boundary_facet[keep_facets]
            self._cells_facets = self.cells('facets')[keep]
            new_index_facets = np.arange(num_facets_old) - np.cumsum(~
                keep_facets)
            self._cells_facets = new_index_facets[self.cells('facets')]
            num_cells_old = len(self.cells('points'))
            new_index_cells = np.arange(num_cells_old) - np.cumsum(~keep)
            ec = self._facets_cells
            ec['boundary'][0] = new_index_facets[ec['boundary'][0]]
            ec['boundary'][1] = new_index_cells[ec['boundary'][1]]
            ec['interior'][0] = new_index_facets[ec['interior'][0]]
            ec['interior'][1:3] = new_index_cells[ec['interior'][1:3]]
            self._facets_cells_idx = None
            self._boundary_facets = None
            self._interior_facets = None
        for k in range(len(self.idx)):
            self.idx[k] = self.idx[k][..., keep]
        if self._volumes is not None:
            for k in range(len(self._volumes)):
                self._volumes[k] = self._volumes[k][..., keep]
        if self._ce_ratios is not None:
            self._ce_ratios = self._ce_ratios[:, keep]
        if self._half_edge_coords is not None:
            self._half_edge_coords = self._half_edge_coords[:, keep]
        if self._ei_dot_ei is not None:
            self._ei_dot_ei = self._ei_dot_ei[:, keep]
        if self._cell_centroids is not None:
            self._cell_centroids = self._cell_centroids[keep]
        if self._circumcenters is not None:
            for k in range(len(self._circumcenters)):
                self._circumcenters[k] = self._circumcenters[k][..., keep, :]
        if self._cell_partitions is not None:
            self._cell_partitions = self._cell_partitions[..., keep]
        if self._signed_cell_volumes is not None:
            self._signed_cell_volumes = self._signed_cell_volumes[keep]
        if self._integral_x is not None:
            self._integral_x = self._integral_x[..., keep, :]
        if self._circumcenter_facet_distances is not None:
            self._circumcenter_facet_distances = (self.
                _circumcenter_facet_distances[..., keep])
        self._signed_circumcenter_distances = None
        self._control_volumes = None
        self._cv_cell_mask = None
        self._cv_centroids = None
        self._cvc_cell_mask = None
        self._is_point_used = None
        self._is_interior_point = None
        self._is_boundary_point = None
        return np.sum(~keep)

    def remove_boundary_cells(self, criterion):
        """Helper method for removing cells along the boundary.

        The input criterion is a callback that must return an array of length

        `sum(mesh.is_boundary_cell)`.



        This helps, for example, in the following scenario.

        When points are moving around, flip_until_delaunay() makes sure the mesh remains

        a Delaunay mesh. This does not work on boundaries where very flat cells can

        still occur or cells may even 'invert'. (The interior point moves outside.) In

        this case, the boundary cell can be removed, and the newly outward node is made

        a boundary node."""
        num_removed = 0
        while True:
            num_boundary_cells = np.sum(self.is_boundary_cell)
            crit = criterion(self.is_boundary_cell)
            if ~np.any(crit):
                break
            if not isinstance(crit, np.ndarray) or crit.shape != (
                num_boundary_cells,):
                raise ValueError(
                    f'criterion() callback must return a Boolean NumPy array of shape {num_boundary_cells,}, got {crit.shape}.'
                    )
            idx = self.is_boundary_cell.copy()
            idx[idx] = crit
            n = self.remove_cells(idx)
            num_removed += n
            if n == 0:
                break
        return num_removed

    def remove_duplicate_cells(self):
        sorted_cells = np.sort(self.cells('points'))
        _, inv, cts = npx.unique_rows(sorted_cells, return_inverse=True,
            return_counts=True)
        remove = np.zeros(len(self.cells('points')), dtype=bool)
        for k in np.where(cts > 1)[0]:
            rem = inv == k
            first_idx = np.where(rem)[0][0]
            rem[first_idx] = False
            remove |= rem
        return self.remove_cells(remove)

    def get_control_volumes(self, cell_mask=None):
        """The control volumes around each vertex. Optionally disregard the

        contributions from particular cells. This is useful, for example, for

        temporarily disregarding flat cells on the boundary when performing Lloyd mesh

        optimization.

        """
        if cell_mask is not None:
            cell_mask = np.asarray(cell_mask)
        if self._cv_centroids is None or np.any(cell_mask != self.
            _cvc_cell_mask):
            if cell_mask is None:
                idx = slice(None)
            else:
                idx = ~cell_mask
            self._control_volumes = npx.sum_at(self.cell_partitions[...,
                idx], self.idx[-1][..., idx], len(self.points))
            self._cv_cell_mask = cell_mask
        assert self._control_volumes is not None
        return self._control_volumes
    control_volumes = property(get_control_volumes)

    @property
    def is_delaunay(self):
        return self.num_delaunay_violations == 0

    @property
    def num_delaunay_violations(self):
        """Number of interior facets where the Delaunay condition is violated."""
        return np.sum(self.signed_circumcenter_distances < 0.0)

    @property
    def idx_hierarchy(self):
        warnings.warn('idx_hierarchy is deprecated, use idx[-1] instead',
            DeprecationWarning)
        return self.idx[-1]

    def show(self, *args, fullscreen=False, **kwargs):
        """Show the mesh (see plot())."""
        import matplotlib.pyplot as plt
        self.plot(*args, **kwargs)
        if fullscreen:
            mng = plt.get_current_fig_manager()
            mng.window.showMaximized()
        plt.show()
        plt.close()

    def save(self, filename, *args, **kwargs):
        """Save the mesh to a file, either as a PNG/SVG or a mesh file"""
        if pathlib.Path(filename).suffix in ['.png', '.svg']:
            import matplotlib.pyplot as plt
            self.plot(*args, **kwargs)
            plt.savefig(filename, transparent=True, bbox_inches='tight')
            plt.close()
        else:
            self.write(filename)

    def plot(self, *args, **kwargs):
        if self.n == 2:
            self._plot_line(*args, **kwargs)
        else:
            assert self.n == 3
            self._plot_tri(*args, **kwargs)

    def _plot_line(self):
        import matplotlib.pyplot as plt
        if len(self.points.shape) == 1:
            x = self.points
            y = np.zeros(self.points.shape[0])
        else:
            assert len(self.points.shape) == 2 and self.points.shape[1] == 2
            x, y = self.points.T
        plt.plot(x, y, '-o')

    def _plot_tri(self, show_coedges=True, control_volume_centroid_color=
        None, mesh_color='k', nondelaunay_edge_color=None,
        boundary_edge_color=None, comesh_color=(0.8, 0.8, 0.8), show_axes=
        True, cell_quality_coloring=None, show_point_numbers=False,
        show_edge_numbers=False, show_cell_numbers=False, cell_mask=None,
        mark_points=None, mark_edges=None, mark_cells=None):
        """Show the mesh using matplotlib."""
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection, PatchCollection
        from matplotlib.patches import Polygon
        fig = plt.figure()
        ax = fig.gca()
        plt.axis('equal')
        if not show_axes:
            ax.set_axis_off()
        xmin = np.amin(self.points[:, 0])
        xmax = np.amax(self.points[:, 0])
        ymin = np.amin(self.points[:, 1])
        ymax = np.amax(self.points[:, 1])
        width = xmax - xmin
        xmin -= 0.1 * width
        xmax += 0.1 * width
        height = ymax - ymin
        ymin -= 0.1 * height
        ymax += 0.1 * height
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        if show_point_numbers:
            for i, x in enumerate(self.points):
                plt.text(x[0], x[1], str(i), bbox={'facecolor': 'w',
                    'alpha': 0.7}, horizontalalignment='center',
                    verticalalignment='center')
        if show_edge_numbers:
            if getattr(self, 'edges', None) is None:
                self.create_facets()
            for i, point_ids in enumerate(self.edges['points']):
                midpoint = np.sum(self.points[point_ids], axis=0) / 2
                plt.text(midpoint[0], midpoint[1], str(i), bbox={
                    'facecolor': 'b', 'alpha': 0.7}, color='w',
                    horizontalalignment='center', verticalalignment='center')
        if show_cell_numbers:
            for i, x in enumerate(self.cell_centroids):
                plt.text(x[0], x[1], str(i), bbox={'facecolor': 'r',
                    'alpha': 0.5}, horizontalalignment='center',
                    verticalalignment='center')
        if cell_quality_coloring:
            cmap, cmin, cmax, show_colorbar = cell_quality_coloring
            plt.tripcolor(self.points[:, 0], self.points[:, 1], self.cells(
                'points'), self.q_radius_ratio, shading='flat', cmap=cmap,
                vmin=cmin, vmax=cmax)
            if show_colorbar:
                plt.colorbar()
        if mark_points is not None:
            idx = mark_points
            plt.plot(self.points[idx, 0], self.points[idx, 1], 'x', color='r')
        if mark_cells is not None:
            if np.asarray(mark_cells).dtype == bool:
                mark_cells = np.where(mark_cells)[0]
            patches = [Polygon(self.points[self.cells('points')[idx]]) for
                idx in mark_cells]
            p = PatchCollection(patches, facecolor='C1')
            ax.add_collection(p)
        if self.edges is None:
            self.create_facets()
        e = self.points[self.edges['points']][:, :, :2]
        if nondelaunay_edge_color is None:
            line_segments0 = LineCollection(e, color=mesh_color)
            ax.add_collection(line_segments0)
        else:
            is_pos = np.zeros(len(self.edges['points']), dtype=bool)
            is_pos[self.interior_facets[self.signed_circumcenter_distances >=
                0]] = True
            is_pos_boundary = self.ce_ratios[self.is_boundary_facet_local] >= 0
            is_pos[self.boundary_facets[is_pos_boundary]] = True
            line_segments0 = LineCollection(e[is_pos], color=mesh_color)
            ax.add_collection(line_segments0)
            line_segments1 = LineCollection(e[~is_pos], color=
                nondelaunay_edge_color)
            ax.add_collection(line_segments1)
        if mark_edges is not None:
            e = self.points[self.edges['points'][mark_edges]][..., :2]
            ax.add_collection(LineCollection(e, color='r'))
        if show_coedges:
            cc = self.cell_circumcenters
            edge_midpoints = 0.5 * (self.points[self.edges['points'][:, 0]] +
                self.points[self.edges['points'][:, 1]])
            a = np.stack([cc[:, :2], edge_midpoints[self.cells('edges')[:, 
                0], :2]], axis=1)
            b = np.stack([cc[:, :2], edge_midpoints[self.cells('edges')[:, 
                1], :2]], axis=1)
            c = np.stack([cc[:, :2], edge_midpoints[self.cells('edges')[:, 
                2], :2]], axis=1)
            line_segments = LineCollection(np.concatenate([a, b, c]), color
                =comesh_color)
            ax.add_collection(line_segments)
        if boundary_edge_color:
            e = self.points[self.edges['points'][self.is_boundary_facet]][:,
                :, :2]
            line_segments1 = LineCollection(e, color=boundary_edge_color)
            ax.add_collection(line_segments1)
        if control_volume_centroid_color is not None:
            centroids = self.get_control_volume_centroids(cell_mask=cell_mask)
            ax.plot(centroids[:, 0], centroids[:, 1], linestyle='', marker=
                '.', color=control_volume_centroid_color)
            for k, centroid in enumerate(centroids):
                plt.text(centroid[0], centroid[1], str(k), bbox=dict(
                    facecolor=control_volume_centroid_color, alpha=0.7),
                    horizontalalignment='center', verticalalignment='center')
        return fig
