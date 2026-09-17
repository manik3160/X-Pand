import { forwardRef, type ButtonHTMLAttributes } from 'react'
import { Slot } from '@radix-ui/react-slot'
import { cva, type VariantProps } from 'class-variance-authority'
import { cn } from '@/lib/utils'

const buttonVariants = cva(
  'inline-flex items-center justify-center gap-2 rounded-xl font-semibold transition-all disabled:opacity-50 disabled:cursor-not-allowed',
  {
    variants: {
      variant: {
        primary: 'bg-accent text-accent-fg hover:brightness-105',
        secondary: 'bg-[rgb(var(--surface-raised)/0.6)] text-text-primary border border-[rgb(var(--line)/0.12)] hover:bg-[rgb(var(--surface-raised)/0.9)]',
        ghost: 'bg-transparent text-text-secondary hover:bg-[rgb(var(--surface-raised)/0.5)] hover:text-text-primary',
        icon: 'bg-transparent text-text-secondary hover:bg-[rgb(var(--surface-raised)/0.6)] hover:text-text-primary p-0',
      },
      size: {
        sm: 'h-8 px-3 text-xs',
        md: 'h-10 px-4 text-sm',
        lg: 'h-12 px-6 text-sm',
        iconSm: 'h-7 w-7',
        iconMd: 'h-9 w-9',
      },
    },
    defaultVariants: { variant: 'secondary', size: 'md' },
  }
)

export interface ButtonProps
  extends ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean
}

const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild = false, ...props }, ref) => {
    const Comp = asChild ? Slot : 'button'
    return <Comp className={cn(buttonVariants({ variant, size }), className)} ref={ref} {...props} />
  }
)
Button.displayName = 'Button'

export default Button
