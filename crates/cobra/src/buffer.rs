use crate::{Context, HeapType};
use ash::vk;
use cobra_bindless::device_pointer::DevicePointer;
use std::num::NonZeroUsize;
use std::ptr::NonNull;
use std::slice::SliceIndex;
use std::{marker::PhantomData, sync::Arc};
use vk_mem::Alloc;

pub struct Buffer<T> {
    context: Arc<Context>,

    pub(crate) buffer: vk::Buffer,
    heap_type: HeapType,
    allocation: vk_mem::Allocation,
    host_address: Option<NonNull<()>>,
    device_address: DevicePointer<T>,
    len: NonZeroUsize,

    _phantom: PhantomData<T>,
}

unsafe impl<T> Send for Buffer<T> {}
unsafe impl<T> Sync for Buffer<T> {}

/// A contiguous buffer that acts like a fixed sized array over `T`
impl<T> Buffer<T> {
    /// # Panics
    /// Will panic if ran out of host or device memory
    pub fn new(context: Arc<Context>, len: NonZeroUsize, heap_type: HeapType) -> Self {
        unsafe {
            let (flags, required_flags) = match heap_type {
                HeapType::Default => (
                    vk_mem::AllocationCreateFlags::empty(),
                    vk::MemoryPropertyFlags::DEVICE_LOCAL,
                ),
                HeapType::HostVisible => (
                    vk_mem::AllocationCreateFlags::MAPPED
                        | vk_mem::AllocationCreateFlags::HOST_ACCESS_RANDOM,
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_CACHED,
                ),
                HeapType::DeviceUpload => (
                    vk_mem::AllocationCreateFlags::MAPPED
                        | vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
                    vk::MemoryPropertyFlags::DEVICE_LOCAL | vk::MemoryPropertyFlags::HOST_VISIBLE,
                ),
            };

            let (buffer, allocation) = context
                .allocator
                .create_buffer(
                    &vk::BufferCreateInfo::default()
                        .size((size_of::<T>() * len.get()) as u64)
                        .usage(
                            vk::BufferUsageFlags::STORAGE_BUFFER
                                | vk::BufferUsageFlags::INDEX_BUFFER
                                | vk::BufferUsageFlags::INDIRECT_BUFFER
                                | vk::BufferUsageFlags::TRANSFER_SRC
                                | vk::BufferUsageFlags::TRANSFER_DST,
                        ),
                    &vk_mem::AllocationCreateInfo {
                        usage: vk_mem::MemoryUsage::Auto,
                        flags,
                        required_flags,
                        ..Default::default()
                    },
                )
                .unwrap();
            let allocation_info = context.allocator.get_allocation_info(&allocation);
            let host_address = (heap_type == HeapType::HostVisible
                || heap_type == HeapType::DeviceUpload)
                .then_some(NonNull::new(allocation_info.mapped_data).unwrap().cast());

            let handle = context.storage_handle.acquire();
            context.device.update_descriptor_sets(
                &[vk::WriteDescriptorSet::default()
                    .dst_set(context.descriptor_set)
                    .dst_binding(crate::STORAGE_BINDING)
                    .dst_array_element(handle)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&[vk::DescriptorBufferInfo::default()
                        .buffer(buffer)
                        .range(vk::WHOLE_SIZE)])],
                &[],
            );

            Buffer {
                context,
                buffer,
                heap_type,
                allocation,
                len,
                host_address,
                device_address: DevicePointer::new(handle),
                _phantom: PhantomData,
            }
        }
    }

    /// Creates a buffer directly from a slice. If `host_local` is set then use [`HeapType::HostVisible`],
    /// otherwise use [`HeapType::DeviceUpload`].
    /// # Panics
    /// If the slice passed in has a length of zero
    pub fn from_slice(context: Arc<Context>, host_local: bool, slice: &[T]) -> Self {
        let buffer = Buffer::new(
            context,
            NonZeroUsize::new(slice.len()).unwrap(),
            if host_local {
                HeapType::HostVisible
            } else {
                HeapType::DeviceUpload
            },
        );
        unsafe {
            slice
                .as_ptr()
                .copy_to(buffer.host_address.unwrap().cast().as_ptr(), slice.len());
        }

        buffer
    }

    /// Returns a reference to an element or subslice depending on the type of index
    /// * If given a position, return a reference to the element at that position
    /// * If given a range, returns the subslice corresponding to that range
    ///
    /// [`None`] is returned if index is out of bounds or if the buffer wasn't created with
    /// [`HeapType::HostVisible`]. For [`HeapType::DeviceUpload`] use [`Buffer::write`]
    /// instead, this function isn't available for that heap type because it'd enable
    /// accidental readback
    pub fn get<I: SliceIndex<[T]>>(&self, index: I) -> Option<&<I as SliceIndex<[T]>>::Output> {
        if self.heap_type != HeapType::HostVisible {
            return None;
        }

        let ptr = unsafe { self.host_address.unwrap_unchecked() };
        let slice = unsafe { std::slice::from_raw_parts(ptr.cast().as_ptr(), self.len()) };
        slice.get(index)
    }

    /// Returns a mutable reference or subslice depending on the type of index,
    /// or [`None`] (see [`Buffer::get`])
    pub fn get_mut<I: SliceIndex<[T]>>(
        &mut self,
        index: I,
    ) -> Option<&mut <I as SliceIndex<[T]>>::Output> {
        if self.heap_type != HeapType::HostVisible {
            return None;
        }

        let ptr = unsafe { self.host_address.unwrap_unchecked() };
        let slice = unsafe { std::slice::from_raw_parts_mut(ptr.cast().as_mut(), self.len()) };
        slice.get_mut(index)
    }

    /// Writes the given slice at the index
    /// # Panics
    /// * Buffer is created with [`HeapType::Default`]
    /// * Slice would be written out of bounds
    pub fn write(&mut self, index: usize, slice: &[T]) {
        assert_ne!(self.heap_type, HeapType::Default);
        assert!((self.len() + index) >= slice.len());

        let ptr = unsafe { self.host_address.unwrap_unchecked() };
        unsafe {
            ptr.cast().add(index).copy_from(
                NonNull::new_unchecked(slice.as_ptr().cast_mut()),
                slice.len(),
            )
        };
    }

    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.len.get()
    }
    pub fn size_bytes(&self) -> usize {
        self.len() * size_of::<T>()
    }
    pub fn device_address(&self) -> DevicePointer<T> {
        self.device_address
    }
}

impl<T> Drop for Buffer<T> {
    fn drop(&mut self) {
        self.context
            .deletion_queue
            .lock()
            .unwrap()
            .push_buffer(self.context.as_ref(), (self.buffer, self.allocation));
        self.context
            .storage_handle
            .recycle(self.device_address.index);
    }
}
