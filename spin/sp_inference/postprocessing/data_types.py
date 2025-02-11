# ====================================== Preliminary Commands =======================================
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, final, ClassVar, Iterable, Optional, Union
from typing_extensions import Self

import numpy as np


# ============================================= Domain ==============================================
@dataclass
class Domain(ABC):
    locations: Any
    label: ClassVar[str] = None
    compatibility_list: ClassVar[Iterable[str]] = None

    @abstractmethod
    def __post_init__(self):
        pass

    @abstractmethod
    def is_compatible_with_data(self, data: np.ndarray) -> None:
        pass

    def is_compatible_with_domain(self, domain: Self) -> bool:
        is_compatible = domain.label in self.compatibility_list
        return is_compatible


# ---------------------------------------------------------------------------------------------------
@final
@dataclass
class Grid1D(Domain):
    locations: np.ndarray
    label: ClassVar[str] = "Grid1D"
    compatibility_list: ClassVar[Iterable[Self]] = ("Grid1D",)

    def __post_init__(self):
        self.locations = np.squeeze(self.locations)
        if not self.locations.ndim == 1:
            raise ValueError(
                f"Grid point array must be 1D, but has dimension {self.locations.ndim}."
            )

    def is_compatible_with_data(self, data: np.ndarray):
        has_matching_dim = data.ndim == 1
        has_matching_size = data.size == self.locations.size
        is_compatible = has_matching_dim and has_matching_size
        return is_compatible


# ---------------------------------------------------------------------------------------------------
@final
@dataclass
class Grid2D(Domain):
    locations: list[np.ndarray]
    label: ClassVar[str] = "Grid2D"
    compatibility_list: ClassVar[Iterable[Self]] = ("Grid2D",)

    def __post_init__(self):
        if not len(self.locations) == 2:
            raise ValueError("2D grid requires two numpy arrays")
        for i, vector in enumerate(self.locations):
            self.locations[i] = np.squeeze(vector)
        if not self.locations[i].ndim == 1:
            raise ValueError(
                f"Grid point arrays must be 1D, "
                f"but array {i} has dimension {self.locations[i].ndim}."
            )

    def is_compatible_with_data(self, data: np.ndarray):
        has_matching_dim = data.ndim == 2
        for i, vector in enumerate(self.locations):
            if not vector.size == data.shape[i]:
                has_matching_size = False
                break
        else:
            has_matching_size = True

        is_compatible = has_matching_dim and has_matching_size
        return is_compatible


# ======================================== Field Components =========================================
@dataclass
class FieldComponent(ABC):
    domain: Domain

    @abstractmethod
    def __post_init__(self):
        pass

    @abstractmethod
    def is_compatible(self, other: Self):
        pass

    @classmethod
    def from_iterable(cls, domain, value_list: Iterable[Any]) -> Iterable[Self]:
        component_list = [cls(domain, values) for values in value_list]
        return component_list


# ---------------------------------------------------------------------------------------------------
@final
@dataclass
class StationaryFieldComponent(FieldComponent):
    values: np.ndarray

    def __post_init__(self):
        self.values = np.squeeze(self.values)
        self.domain.is_compatible_with_data(self.values)

    def is_compatible(self, other: Self):
        is_compatible = self.domain.is_compatible_with_domain(other.domain)
        return is_compatible


# ---------------------------------------------------------------------------------------------------
@final
@dataclass
class TimeDependentFieldComponent(FieldComponent):
    times: np.ndarray
    values: Iterable[np.ndarray]

    def __post_init__(self):
        self.times = np.squeeze(self.times)
        if not self.times.ndim == 1:
            raise ValueError(
                f"Time array must be 1D, but has dimension {self.times.ndim}."
            )
        if not self.times.size == len(self.values):
            raise ValueError(
                f"Number of time points ({self.times.size}) "
                f"does not match number of value arrays ({len(self.values)})."
            )
        if self.times.size < 3:
            raise ValueError(
                f"Number of time points ({self.times.size}) needs to be at least 3."
            )

        for i, value_array in enumerate(self.values):
            self.values[i] = np.squeeze(value_array)
            if not self.domain.is_compatible_with_data(self.values[i]):
                raise ValueError(
                    f"Values for time point {i} are not compatible with domain."
                )

    def is_compatible(self, other: Self):
        domains_compatible = self.domain.is_compatible_with_domain(other.domain)
        times_compatible = self.times.size == other.times.size
        is_compatible = domains_compatible and times_compatible
        return is_compatible


# ============================================= Fields ==============================================
@dataclass
class Field:
    components: Union[FieldComponent, Iterable[FieldComponent]]
    num_components: int = field(init=False)

    def __post_init__(self):
        if isinstance(self.components, FieldComponent):
            self.components = (self.components,)
        self.num_components = len(self.components)

    def is_compatible(self, other: Self):
        compatible_component_num = self.num_components == other.num_components
        for i, component in enumerate(self.components):
            if not component.is_compatible(other.components[i]):
                compatible_components = False
                break
        else:
            compatible_components = True

        is_compatible = compatible_components and compatible_component_num
        return is_compatible


# ---------------------------------------------------------------------------------------------------
@final
@dataclass
class StationaryField(Field):
    components: Iterable[StationaryFieldComponent]


@final
@dataclass
class TimeDependentField(Field):
    components: Iterable[TimeDependentFieldComponent]


# ========================================= Inference Data ==========================================
@dataclass
class InferenceData:
    label: str
    field: Field
    variance_field: Optional[Field] = None
    is_point_data: Optional[bool] = False

    def __post_init__(self):
        if self.variance_field:
            if not isinstance(self.variance_field, type(self.field)):
                raise ValueError("Variance field must be of same type as mean field.")
            if not self.field.is_compatible(self.variance_field):
                raise ValueError("Mean and variance fields must be of equal size.")

    def is_compatible(self, other: Self):
        return self.field.is_compatible(other.field)
