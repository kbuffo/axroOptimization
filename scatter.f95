subroutine primaryPSF(distortion,xdim,ydim,z,length,x0,N,wave,primfoc,r0,graze,psf)
  !Declarations
  implicit none
  integer, intent(in) :: xdim,ydim,N
  real*8, intent(in), dimension(xdim,ydim) :: distortion
  real*8, intent(in), dimension(ydim) :: z
  real*8, intent(in), dimension(N) :: x0
  real*8, intent(in), dimension(xdim) :: length
  real*8, intent(in) :: wave,primfoc,r0,graze
  complex*16 :: j,integrand
  real*8, intent(out), dimension(N) :: psf
  integer :: i,xi,yi
  real*8 :: d2,lengthsum,dz,dummy,pi

  !Constants
  j = (0.,1.)
  lengthsum = sum(length)
  dz = abs(z(2)-z(1))
  pi = acos(-1.)

  do i = 1,N
    psf(i) = 0.
    do xi = 1,xdim
      !Compute integral over axial dimension
      integrand = 0.
      !$omp parallel do reduction(+:integrand) private(d2)
      do yi = 1,ydim
        if (isnan(distortion(xi,yi)) .eqv. .False.) then
          d2 = sqrt((distortion(xi,yi)-x0(i))**2 + (z(yi)+primfoc)**2)
          integrand = integrand + exp(-2*j*pi*(d2-z(yi))/wave)*sqrt(distortion(xi,yi)/d2)
          !print *, distortion(xi,yi), z(yi), d2, integrand
          !read *, dummy
        end if
      end do
      !$omp end parallel do
      !Add weighted integral to relevant observation point
      !print *, integrand,abs(integrand)**2,x0(i)
      psf(i) = psf(i) + abs(integrand)**2*sin(graze)*dz**2/wave/r0/lengthsum
      !print *, psf(i)
      !read *, dummy
    end do

  end do

end subroutine primaryPSF
