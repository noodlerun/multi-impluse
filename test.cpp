#pragma once

#include <math.h>

template <typename T> struct MATRIX33;

template <typename T>
struct VECTOR3
{
public:
    inline VECTOR3() {};
    inline VECTOR3(const T*);
    inline VECTOR3(T x, T y, T z);

    friend inline VECTOR3<T> operator * (T f, const VECTOR3<T>& v)
    {
        return VECTOR3<T>(f * v.x, f * v.y, f * v.z);
    }
    ;

    // casting
    operator T* ();
    operator const T* () const;

    template <typename T1> explicit operator VECTOR3<T1>();


    // assignment operators
    VECTOR3<T>& operator += (const VECTOR3<T>&);
    VECTOR3<T>& operator -= (const VECTOR3<T>&);
    VECTOR3<T>& operator *= (T);
    VECTOR3<T>& operator *= (const VECTOR3<T>&);
    VECTOR3<T>& operator *= (const MATRIX33<T>&);
    VECTOR3<T>& operator /= (T);
    VECTOR3<T>& operator /= (const VECTOR3<T>&);

    // unary operators
    VECTOR3<T> operator + () const;
    VECTOR3<T> operator - () const;

    // binary operators
    VECTOR3<T> operator + (const VECTOR3<T>&) const;
    VECTOR3<T> operator - (const VECTOR3<T>&) const;
    VECTOR3<T> operator * (T) const;
    VECTOR3<T> operator * (const VECTOR3<T>&) const;
    VECTOR3<T> operator * (const MATRIX33<T>&) const;
    VECTOR3<T> operator / (T) const;
    VECTOR3<T> operator / (const VECTOR3<T>&) const;


    bool operator == (const VECTOR3<T>&) const;
    bool operator != (const VECTOR3<T>&) const;

    union    {
        struct  {
            T x, y, z;
        };
        T v[3];
    };

};


template <typename T>
inline VECTOR3<T>::VECTOR3(const T* pf)
{
    x = pf[0];
    y = pf[1];
    z = pf[2];
}

template <typename T>
inline VECTOR3<T>::VECTOR3(T fx, T fy, T fz)
{
    x = fx;
    y = fy;
    z = fz;
}

template<typename T>
template<typename T1>
inline VECTOR3<T>::operator VECTOR3<T1>()
{
    return VECTOR3<T1>((T1)x, (T1)y, (T1)z);
}

template <typename T>
inline VECTOR3<T>::operator T* ()
{
    return (T*)&x;
}

template <typename T>
inline VECTOR3<T>::operator const T* () const
{
    return (const T*)&x;
}

template <typename T>
inline VECTOR3<T>& VECTOR3<T>::operator += (const VECTOR3<T>& v)
{
    x += v.x;
    y += v.y;
    z += v.z;
    return *this;
}

template <typename T>
inline VECTOR3<T>& VECTOR3<T>::operator -= (const VECTOR3<T>& v)
{
    x -= v.x;
    y -= v.y;
    z -= v.z;
    return *this;
}

template <typename T>
inline VECTOR3<T>& VECTOR3<T>::operator *= (T f)
{
    x *= f;
    y *= f;
    z *= f;
    return *this;
}

template<typename T>
inline VECTOR3<T>& VECTOR3<T>::operator*=(const VECTOR3<T>& v)
{
    x *= v.x;
    y *= v.y;
    z *= v.z;
    return *this;
}

template<typename T>
inline VECTOR3<T>& VECTOR3<T>::operator*=(const MATRIX33<T>& mt)
{
    x = x * mt._11 + y * mt._21 + z * mt._31;
    y = x * mt._12 + y * mt._22 + z * mt._32;
    z = x * mt._13 + y * mt._23 + z * mt._33;

    return *this;
}

template <typename T>
inline VECTOR3<T>& VECTOR3<T>::operator /= (T f)
{
    x /= f;
    y /= f;
    z /= f;
    return *this;
}

template<typename T>
inline VECTOR3<T>& VECTOR3<T>::operator/=(const VECTOR3<T>& v)
{
    x /= v.x;
    y /= v.y;
    z /= v.z;
    return *this;
}

template <typename T>
inline VECTOR3<T> VECTOR3<T>::operator + () const
{
    return *this;
}

template <typename T>
inline VECTOR3<T> VECTOR3<T>::operator - () const
{
    return VECTOR3<T>(-x, -y, -z);
}

template <typename T>
inline VECTOR3<T> VECTOR3<T>::operator + (const VECTOR3<T>& v) const
{
    return VECTOR3<T>(x + v.x, y + v.y, z + v.z);
}

template <typename T>
inline VECTOR3<T> VECTOR3<T>::operator - (const VECTOR3<T>& v) const
{
    return VECTOR3<T>(x - v.x, y - v.y, z - v.z);
}

template <typename T>
inline VECTOR3<T> VECTOR3<T>::operator * (T f) const
{
    return VECTOR3<T>(x * f, y * f, z * f);
}

template<typename T>
inline VECTOR3<T> VECTOR3<T>::operator * (const VECTOR3<T>& v) const
{
    return VECTOR3<T>(x * v.x, y * v.y, z * v.z);
}

template<typename T>
inline VECTOR3<T> VECTOR3<T>::operator * (const MATRIX33<T>& mt) const
{
    return VECTOR3<T>(x * mt._11 + y * mt._21 + z * mt._31,
        x * mt._12 + y * mt._22 + z * mt._32,
        x * mt._13 + y * mt._23 + z * mt._33);
}

template <typename T>
inline VECTOR3<T> VECTOR3<T>::operator / (T f) const
{
    return VECTOR3<T>(x / f, y / f, z / f);
}

template<typename T>
inline VECTOR3<T> VECTOR3<T>::operator / (const VECTOR3<T>& v) const
{
    return VECTOR3<T>(x / v.x, y / v.y, z / v.z);
}

template <typename T>
inline bool VECTOR3<T>::operator == (const VECTOR3<T>& v) const
{
    return x == v.x && y == v.y && z == v.z;
}

template <typename T>
inline bool VECTOR3<T>::operator != (const VECTOR3<T>& v) const
{
    return x != v.x || y != v.y || z != v.z;
}


template <typename T>
inline VECTOR3<T> round(VECTOR3<T> vec3)
{
    return VECTOR3<T>(round(vec3.x), round(vec3.y), round(vec3.z));
}

template <typename T>
inline VECTOR3<T> pow(VECTOR3<T> v1, T dwPow)
{
    return VECTOR3<T>(pow(v1.x, dwPow), pow(v1.y, dwPow), pow(v1.z, dwPow));
}


template <typename T>
inline VECTOR3<T> abs(VECTOR3<T> v)
{
    return VECTOR3<T>(abs(v.x), abs(v.y), abs(v.z));
}

template <typename T>
inline VECTOR3<T> ceil(VECTOR3<T> v)
{
    return VECTOR3<T>(ceil(v.x), ceil(v.y), ceil(v.z));
}

template <typename T>
inline VECTOR3<T> floor(VECTOR3<T> v)
{
    return VECTOR3<T>(floor(v.x), floor(v.y), floor(v.z));
}

template <typename T>
inline VECTOR3<T> Max(VECTOR3<T> a, VECTOR3<T> b)
{
    return VECTOR3<T>(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z));
}

template <typename T>
inline VECTOR3<T> Max(VECTOR3<T> v)
{
    return VECTOR3<T>(max(max(v.x, v.y), v.z));
}

template <typename T>
inline VECTOR3<T> Min(VECTOR3<T> a, VECTOR3<T> b)
{
    return VECTOR3<T>(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z));
}

template <typename T>
inline VECTOR3<T> Min(VECTOR3<T> v)
{
    return VECTOR3<T>(min(min(v.x, v.y), v.z));
}

template<typename T>
inline T len(VECTOR3<T> v)
{
    return sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

template<typename T>
inline VECTOR3<T> normalize(VECTOR3<T> v)
{
    T Size = sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    return VECTOR3<T>(v.x / Size, v.y / Size, v.z / Size);
}

template<typename T>
inline VECTOR3<T> project(VECTOR3<T> a, VECTOR3<T> b)
{
    return b * (dot(a, b) / dot(b, b));
}

template<typename T>
inline VECTOR3<T> cross(VECTOR3<T> s1, VECTOR3<T> s2)
{
    return VECTOR3<T>(s1.y * s2.z - s1.z * s2.y,
        s1.z * s2.x - s1.x * s2.z,
        s1.x * s2.y - s1.y * s2.x);
}

template <typename T>
inline T dot(VECTOR3<T> s1, VECTOR3<T> s2)
{
    return (s1.x * s2.x + s1.y * s2.y + s1.z * s2.z);
}

template<typename T>
inline VECTOR3<T> reflect(VECTOR3<T> i, VECTOR3<T> n)
{
    return i - 2 * dot(i, n) * n;
}

template<int Row, int Col, typename T = float>
struct MATRIX
{
    MATRIX()
    {
        for (int i = 0; i < Row; i++)
            for (int j = 0; j < Col; j++)
                aNumbers[i][j] = 0;
    }

    T operator ()(int dwRow, int dwCol) const
    {
        return aNumbers[dwRow][dwCol];
    }

    T& operator ()(int dwRow, int dwCol)
    {
        return aNumbers[dwRow][dwCol];
    }

    template<int MulRow, int MulCol>
    MATRIX<Row, MulCol, T> operator * (const MATRIX<MulRow, MulCol, T>& m) const
    {
        MATRIX<Row, MulCol, T> result;

        if constexpr (Col != MulRow)
            static_assert("error");

        for (int row_out = 0; row_out < Row; row_out++)
            for (int col_out = 0; col_out < MulCol; col_out++)
                for (int row = 0; row < MulRow; row++)
                    result(row_out, col_out) += (aNumbers[row_out][row] * m(row, col_out));

        return result;

    };

    inline MATRIX<Row, Col, T> operator - (const MATRIX<Row, Col, T>& m) const
    {
        MATRIX<Row, Col, T> result;
        for (int row_out = 0; row_out < Row; row_out++)
            for (int col_out = 0; col_out < Col; col_out++)
                result(row_out, col_out) = (aNumbers[row_out][col_out] - m(row_out, col_out));

        return result;
   }

    inline MATRIX<Row, Col, T> operator - () const
    {
        MATRIX<Row, Col, T> result;
        for (int row_out = 0; row_out < Row; row_out++)
            for (int col_out = 0; col_out < Col; col_out++)
                result(row_out, col_out) = -aNumbers[row_out][col_out];

        return result;
    }


    inline MATRIX<Row, Col, T> operator * (T scalar)
    {
        MATRIX<Row, Col, T> result;
        for (int row_out = 0; row_out < Row; row_out++)
            for (int col_out = 0; col_out < Col; col_out++)
                result.aNumbers[row_out][col_out] = aNumbers[row_out][col_out] * scalar;

        return result;
    }


    inline MATRIX<Col, Row, T> transpose()
    {
        MATRIX<Col, Row, T> result;
        for (int row_out = 0; row_out < Col; row_out++)
            for (int col_out = 0; col_out < Row; col_out++)
                result(row_out, col_out) = aNumbers[col_out][row_out];

        return result;
    }

    T aNumbers[Row][Col];
};

template<typename T = float>
MATRIX<3, 3, T>	inverse(const MATRIX<3, 3, T>& M)
{
    T c00 = M(1, 1) * M(2, 2) - M(1, 2) * M(2, 1);
    T c10 = M(1, 2) * M(2, 0) - M(1, 0) * M(2, 2);
    T c20 = M(1, 0) * M(2, 1) - M(1, 1) * M(2, 0);
    T det = M(0, 0) * c00 + M(0, 1) * c10 + M(0, 2) * c20;
    if (det != (T)0)
    {
        MATRIX<3, 3, T> result;
        T invDet = ((T)1) / det;

        result(0, 0) = c00 * invDet;
        result(0, 1) = (M(0, 2) * M(2, 1) - M(0, 1) * M(2, 2)) * invDet;
        result(0, 2) = (M(0, 1) * M(1, 2) - M(0, 2) * M(1, 1)) * invDet;
        result(1, 0) = c10 * invDet;
        result(1, 1) = (M(0, 0) * M(2, 2) - M(0, 2) * M(2, 0)) * invDet;
        result(1, 2) = (M(0, 2) * M(1, 0) - M(0, 0) * M(1, 2)) * invDet;
        result(2, 0) = c20 * invDet;
        result(2, 1) = (M(0, 1) * M(2, 0) - M(0, 0) * M(2, 1)) * invDet;
        result(2, 2) = (M(0, 0) * M(1, 1) - M(0, 1) * M(1, 0)) * invDet;

        return result;
    }
    else
    {
        return MATRIX<3, 3, T>();
    }
}

template<typename T>
MATRIX<3, 3, T> GetTensorOfInertiaOfCuboid(const T& fMass, const VECTOR3<T>& vSize)
{
    MATRIX<3, 3, T> result;

    result(0, 0) = vSize.y * vSize.y + vSize.z * vSize.z;
    result(0, 1) = 0;
    result(0, 2) = 0;
    result(1, 0) = 0;
    result(1, 1) = vSize.x * vSize.x + vSize.z * vSize.z;
    result(1, 2) = 0;
    result(2, 0) = 0;
    result(2, 1) = 0;
    result(2, 2) = vSize.x * vSize.x + vSize.y * vSize.y;

    return result * (fMass / 12);
};

template<typename T>
struct VelocityConstraintParameter
{
    T				fInverseMass1;
    MATRIX<3,3,T>	mtInverseInertia1;
    VECTOR3<T>		vBarycenter1;
    VECTOR3<float>  vVelocity1;
    VECTOR3<float>  vAngularVelocity1;

    T				fInverseMass2;
    MATRIX<3, 3, T>	mtInverseInertia2;
    VECTOR3<T>		vBarycenter2;
    VECTOR3<float>  vVelocity2;
    VECTOR3<float>  vAngularVelocity2;

    VECTOR3<T>						vObj1ContactPoint;
    VECTOR3<T>						vObj2ContactPoint;
    VECTOR3<T>						vDirection;
    T							    RestitutionCoefficient;

    VECTOR3<float>* pOutVelocity1;
    VECTOR3<float>* pOutAngularVelocity1;
    VECTOR3<float>* pOutVelocity2;
    VECTOR3<float>* pOutAngularVelocity2;
};

template<typename T>
void VelocityMultipleConstraint(const VelocityConstraintParameter<T>* param)
{
    MATRIX<3, 12, double> Jacobian;
    MATRIX<12, 12, double> InverseMass;
    MATRIX<12, 1, double> Velocity;

    for (int i = 0; i < 3; i++) {
        Jacobian(i, 0) = param[i].vDirection.x;
        Jacobian(i, 1) = param[i].vDirection.y;
        Jacobian(i, 2) = param[i].vDirection.z;

        const VECTOR3<T> vRadiusObject1 = param[i].vObj1ContactPoint - param[i].vBarycenter1;
        const VECTOR3<T> vRadiusCrossNormal1 = cross(vRadiusObject1, param[i].vDirection);

        Jacobian(i, 3) = vRadiusCrossNormal1.x;
        Jacobian(i, 4) = vRadiusCrossNormal1.y;
        Jacobian(i, 5) = vRadiusCrossNormal1.z;

        Jacobian(i, 6) = -param[i].vDirection.x;
        Jacobian(i, 7) = -param[i].vDirection.y;
        Jacobian(i, 8) = -param[i].vDirection.z;

        const VECTOR3<T> vRadiusObject2 = param[i].vObj2ContactPoint - param[i].vBarycenter2;
        const VECTOR3<T> vRadiusCrossNormal2 = cross(-vRadiusObject2, param[i].vDirection);

        Jacobian(i, 9) = vRadiusCrossNormal2.x;
        Jacobian(i, 10) = vRadiusCrossNormal2.y;
        Jacobian(i, 11) = vRadiusCrossNormal2.z;
    }
    InverseMass(0, 0) = param[0].fInverseMass1;             InverseMass(1, 1) = param[0].fInverseMass1;                 InverseMass(2, 2) = param[0].fInverseMass1;
    InverseMass(3, 3) = param[0].mtInverseInertia1(0, 0);   InverseMass(3, 4) = param[0].mtInverseInertia1(0, 1);       InverseMass(3, 5) = param[0].mtInverseInertia1(0, 2);
    InverseMass(4, 3) = param[0].mtInverseInertia1(1, 0);   InverseMass(4, 4) = param[0].mtInverseInertia1(1, 1);       InverseMass(4, 5) = param[0].mtInverseInertia1(1, 2);
    InverseMass(5, 3) = param[0].mtInverseInertia1(2, 0);   InverseMass(5, 4) = param[0].mtInverseInertia1(2, 1);       InverseMass(5, 5) = param[0].mtInverseInertia1(2, 2);


    InverseMass(6, 6) = param[0].fInverseMass2;               InverseMass(7, 7) = param[0].fInverseMass2;               InverseMass(8, 8) = param[0].fInverseMass2;
    InverseMass(9, 9) = param[0].mtInverseInertia2(0, 0);     InverseMass(9, 10) = param[0].mtInverseInertia2(0, 1);     InverseMass(9, 11) = param[0].mtInverseInertia2(0, 2);
    InverseMass(10, 9) = param[0].mtInverseInertia2(1, 0);     InverseMass(10, 10) = param[0].mtInverseInertia2(1, 1);     InverseMass(10, 11) = param[0].mtInverseInertia2(1, 2);
    InverseMass(11, 9) = param[0].mtInverseInertia2(2, 0);     InverseMass(11, 10) = param[0].mtInverseInertia2(2, 1);     InverseMass(11, 11) = param[0].mtInverseInertia2(2, 2);

    Velocity(0, 0) = param[0].vVelocity1.x;         Velocity(1, 0) = param[0].vVelocity1.y;            Velocity(2, 0) = param[0].vVelocity1.z;
    Velocity(3, 0) = param[0].vAngularVelocity1.x;  Velocity(4, 0) = param[0].vAngularVelocity1.y;     Velocity(5, 0) = param[0].vAngularVelocity1.z;
    Velocity(6, 0) = param[0].vVelocity2.x;         Velocity(7, 0) = param[0].vVelocity2.y;            Velocity(8, 0) = param[0].vVelocity2.z;
    Velocity(9, 0) = param[0].vAngularVelocity2.x;  Velocity(10, 0) = param[0].vAngularVelocity2.y;     Velocity(11, 0) = param[0].vAngularVelocity2.z;

    MATRIX<3, 1, double> Lambda = inverse( Jacobian * InverseMass * Jacobian.transpose()) * (-Jacobian * Velocity);

    MATRIX<12, 1, double> DeltaV = InverseMass * Jacobian.transpose() * Lambda;

    *param[0].pOutVelocity1          = VECTOR3<float>(param[0].vVelocity1.x + DeltaV(0, 0), param[0].vVelocity1.y + DeltaV(1, 0), param[0].vVelocity1.z + DeltaV(2, 0));
    *param[0].pOutAngularVelocity1   = VECTOR3<float>(param[0].vAngularVelocity1.x + DeltaV(3, 0), param[0].vAngularVelocity1.y + DeltaV(4, 0), param[0].vAngularVelocity1.z + DeltaV(5, 0));
    *param[0].pOutVelocity2          = VECTOR3<float>(param[0].vVelocity2.x + DeltaV(6, 0), param[0].vVelocity2.y + DeltaV(7, 0), param[0].vVelocity2.z + DeltaV(8, 0));
    *param[0].pOutAngularVelocity2   = VECTOR3<float>(param[0].vAngularVelocity2.x + DeltaV(9, 0), param[0].vAngularVelocity2.y + DeltaV(10, 0), param[0].vAngularVelocity2.z + DeltaV(11, 0));
   
}


template<typename T>
VECTOR3<T> CalcuPointVelocity(VECTOR3<float> vVelocity, VECTOR3<float> vAngularVelocity, const VECTOR3<float>& vCenter, VECTOR3<T> vPoint)
{
    const VECTOR3<T> vRadiusObject = vPoint - vCenter;
    return vVelocity + cross(vAngularVelocity, vRadiusObject);
}

int main() {
    float fMass1 = 100;
    float fMass2 = 100;
    const VECTOR3<float> vCenter[] = { VECTOR3<float>(0, 9, 0), VECTOR3<float>(0, 0, 0) };
    const VECTOR3<float> ContactPoint[] = { VECTOR3<float>(-3, 4.5, -3), VECTOR3<float>(-3, 4.5, 3), VECTOR3<float>(3, 4.5, -3), VECTOR3<float>(3, 4.5, 3) };
    const VECTOR3<float> vSize = VECTOR3<float>(6, 9, 6);

    VECTOR3<float> vVelocity1           = VECTOR3<float>(0, 20, 0);
    VECTOR3<float> vAngularVelocity1    = VECTOR3<float>(0, 0, -3);

    VECTOR3<float> vVelocity2           = VECTOR3<float>(0, 80, 0);
    VECTOR3<float> vAngularVelocity2    = VECTOR3<float>(10, 0, 0);

    VECTOR3<float> vResultVelocity1;
    VECTOR3<float> vResultAngularVelocity1;
    VECTOR3<float> vResultVelocity2;
    VECTOR3<float> vResultAngularVelocity2;
    VelocityConstraintParameter<float> param[3];
    for (int i = 0; i < 3; i++) {

        param[i].pOutVelocity1          = &vResultVelocity1;
        param[i].pOutAngularVelocity1   = &vResultAngularVelocity1;
        param[i].pOutVelocity2          = &vResultVelocity2;
        param[i].pOutAngularVelocity2   = &vResultAngularVelocity2;

        param[i].RestitutionCoefficient = 0;

        param[i].fInverseMass1      = 1 / fMass1;
        param[i].vBarycenter1       = vCenter[0];
        param[i].vVelocity1         = vVelocity1;
        param[i].vAngularVelocity1  = vAngularVelocity1;

        param[i].fInverseMass2      = 1.0f / fMass2;
        param[i].vBarycenter2       = vCenter[1];
        param[i].vVelocity2         = vVelocity2;
        param[i].vAngularVelocity2  = vAngularVelocity2;

        param[i].mtInverseInertia1 = inverse(GetTensorOfInertiaOfCuboid(fMass1, vSize));
        param[i].mtInverseInertia2 = inverse(GetTensorOfInertiaOfCuboid(fMass2, vSize));

        VECTOR3<float> vPointVelocity1 = CalcuPointVelocity(vVelocity1, vAngularVelocity1, param[i].vBarycenter1, ContactPoint[i]);
        VECTOR3<float> vPointVelocity2 = CalcuPointVelocity(vVelocity2, vAngularVelocity2, param[i].vBarycenter2, ContactPoint[i]);

        param[i].vObj1ContactPoint = ContactPoint[i] + vPointVelocity1 * 0.0001;
        param[i].vObj2ContactPoint = ContactPoint[i] + vPointVelocity2 * 0.0001;

        param[i].vDirection = normalize(param[i].vObj1ContactPoint - param[i].vObj2ContactPoint);
    }
    
    VelocityMultipleConstraint<float>(param);

    VECTOR3<float> vPointVelocity11 = CalcuPointVelocity(vResultVelocity1, vResultAngularVelocity1, param[0].vBarycenter1, ContactPoint[0]);
    VECTOR3<float> vPointVelocity21 = CalcuPointVelocity(vResultVelocity2, vResultAngularVelocity2, param[0].vBarycenter2, ContactPoint[0]);

    VECTOR3<float> vPointVelocity12 = CalcuPointVelocity(vResultVelocity1, vResultAngularVelocity1, param[0].vBarycenter1, ContactPoint[1]);
    VECTOR3<float> vPointVelocity22 = CalcuPointVelocity(vResultVelocity2, vResultAngularVelocity2, param[0].vBarycenter2, ContactPoint[1]);

    VECTOR3<float> vPointVelocity13 = CalcuPointVelocity(vResultVelocity1, vResultAngularVelocity1, param[0].vBarycenter1, ContactPoint[2]);
    VECTOR3<float> vPointVelocity23 = CalcuPointVelocity(vResultVelocity2, vResultAngularVelocity2, param[0].vBarycenter2, ContactPoint[2]);

    return 0;
}