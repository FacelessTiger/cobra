use crate::Descriptors;
use core::marker::PhantomData;
use spirv_std::ByteAddressableBuffer;
use spirv_std::macros::gpu_only;

/// Emulates a device pointer through an index into a bindless set of buffers + byte offset into the buffer at that index.
/// Made to mimic the semantics of [`core::ptr::NonNull`]. Isn't usable until [`DevicePointer::validate`] is
/// called, and it's converted into a [`PhysicalPointer`] shader side.
///
/// This is a temporary solution until [buffer device addresses](https://github.com/Rust-GPU/rust-gpu/pull/237) are
/// merged into rust-gpu
#[repr(C)]
pub struct DevicePointer<T> {
    #[allow(unused)]
    pub index: u32,
    byte_offset: u32,
    _phantom: PhantomData<T>,
}

impl<T> Clone for DevicePointer<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for DevicePointer<T> {}

impl<T> DevicePointer<T> {
    pub fn new(handle: u32) -> DevicePointer<T> {
        DevicePointer {
            index: handle,
            byte_offset: 0,
            _phantom: PhantomData,
        }
    }

    pub fn cast<U>(self) -> DevicePointer<U> {
        DevicePointer {
            index: self.index,
            byte_offset: self.byte_offset,
            _phantom: PhantomData,
        }
    }

    #[gpu_only]
    pub fn validate<'a>(&self, descriptors: &'a Descriptors) -> PhysicalPointer<'a, T> {
        PhysicalPointer {
            byte_addressable_buffer: ByteAddressableBuffer::from_slice(unsafe {
                descriptors.buffers.index(self.index as usize)
            }),
            byte_offset: self.byte_offset,
            _phantom: PhantomData,
        }
    }
}

pub struct PhysicalPointer<'a, T> {
    byte_addressable_buffer: ByteAddressableBuffer<&'a [u32]>,
    byte_offset: u32,
    _phantom: PhantomData<T>,
}

impl<'a, T> PhysicalPointer<'a, T> {
    pub fn cast<U>(self) -> PhysicalPointer<'a, U> {
        PhysicalPointer {
            byte_addressable_buffer: self.byte_addressable_buffer,
            byte_offset: self.byte_offset,
            _phantom: PhantomData,
        }
    }

    /// Deref the pointer, really loads the buffer at the internal byte offset.
    /// # Safety
    /// There are no bounds or type checks, any out of bound deref/writes or on
    /// a wrong type is undefined behavior.
    pub unsafe fn deref(&self) -> T {
        unsafe { self.byte_addressable_buffer.load(self.byte_offset) }
    }

    /*/// Write to the pointer, really loads the buffer at the internal byte offset.
    /// # Safety
    /// There are no bounds or type checks, any out of bound deref/writes or on
    /// a wrong type is undefined behavior.
    pub unsafe fn write(&mut self, value: T) {
        self.byte_addressable_buffer.store(self.byte_offset, value);
    }*/
}

trait PointerGet {
    fn consume_self(self, byte_offset: u32) -> Self;
    fn get_byte_offset(&self) -> u32;
}

#[allow(private_bounds)]
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
#[allow(clippy::cast_possible_wrap)]
pub trait PointerType<T>: PointerGet
where
    Self: Sized,
{
    // These 'allows' are necessary because `try_into` uses `Result`
    // and value enums don't work in rust-gpu yet

    #[must_use]
    fn offset(self, count: isize) -> Self {
        let byte_offset = self.get_byte_offset();
        self.consume_self(byte_offset + (count * size_of::<T>() as isize) as u32)
    }

    #[must_use]
    fn add(self, count: usize) -> Self {
        let byte_offset = self.get_byte_offset();
        self.consume_self(byte_offset + (count * size_of::<T>()) as u32)
    }

    #[must_use]
    fn byte_offset(self, offset: isize) -> Self {
        let byte_offset = self.get_byte_offset();
        self.consume_self((byte_offset as isize + offset) as u32)
    }
}

impl<T> PointerGet for DevicePointer<T> {
    fn consume_self(self, byte_offset: u32) -> Self {
        DevicePointer {
            byte_offset,
            ..self
        }
    }
    fn get_byte_offset(&self) -> u32 {
        self.byte_offset
    }
}
impl<T> PointerGet for PhysicalPointer<'_, T> {
    fn consume_self(self, byte_offset: u32) -> Self {
        PhysicalPointer {
            byte_offset,
            ..self
        }
    }
    fn get_byte_offset(&self) -> u32 {
        self.byte_offset
    }
}

impl<T> PointerType<T> for DevicePointer<T> {}
impl<T> PointerType<T> for PhysicalPointer<'_, T> {}
